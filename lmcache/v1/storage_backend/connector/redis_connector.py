# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from enum import IntEnum, auto
from typing import List, Optional, Tuple, no_type_check
import asyncio
import inspect
import os
import queue
import socket
import threading

# Third Party
from redis.asyncio.cluster import ClusterNode, RedisCluster

# note: asyncio.wrap_future preserves the synchronization property of asyncio.Future
# but loses the concurrency property on thread-level blocking behavior
import redis.asyncio as redis

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class RESPClient:
    """
    A client implementing RESP2 only for GET, SET, and EXISTS
    Should be wrapped with MultiRESPClient

    Primary Assumption (for "chunked" parsing and reusing payloads):
    The size of payloads (KV cache object) is always fixed. The retrieval
    helper `_recv_exactly(n, buf)` can be used to retrieve payloads without
    having to scan for \r\n (`save_unfull_chunk` should be False)


    Optimizations:
    - zero copy retrieval (through recv_into) ** not supported by redis-py **
    - scatter-gather sending (through sendmsg)
    """

    def __init__(self, host: str, port: int, chunk_size: int):
        """
        the chunk_size must be known beforehand (save_unfull_chunk = False)
        for this client to work
        """
        self.chunk_size = chunk_size
        self._generate_reusables(chunk_size)
        self.sock = socket.create_connection((host, port))

        # Optimize socket for low-latency bulk transfers
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Increase socket buffers for 4MB chunks
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)

    def _generate_reusables(self, chunk_size: int):
        # some cached objects for scatter-gather sending
        # and response parsing
        self.size_header = memoryview(f"${chunk_size}\r\n".encode())
        self.size_header_len = len(self.size_header)

        self.crlf = memoryview(b"\r\n")
        self.crlf_len = len(self.crlf)

        self._get_prefix = [
            memoryview(b"*2\r\n"),
            memoryview(b"$3\r\nGET\r\n"),
        ]

        self._set_prefix = [
            memoryview(b"*3\r\n"),
            memoryview(b"$3\r\nSET\r\n"),
        ]

        self._exists_prefix = [
            memoryview(b"*2\r\n"),
            memoryview(b"$6\r\nEXISTS\r\n"),
        ]

        # simple string response for set
        self._ok = memoryview(b"+OK\r\n")
        self._ok_len = len(self._ok)

        # integer response for exists
        self._one = memoryview(b":1\r\n")
        self._zero = memoryview(b":0\r\n")
        # assumes int < 256
        self._int_len = len(self._one)  # len(self._zero)

    # -- recv and send (optimized for zero copy) ---

    def _recv_exactly_into(self, n: int, into: memoryview):
        """
        Reads exactly n bytes.
        """
        assert into is not None
        total = 0
        while total < n:
            m = self.sock.recv_into(into[total:n])
            if m == 0:
                raise ConnectionError("Socket closed during recv_exactly")
            total += m

    def _send_multipart(self, parts: list[memoryview]):
        """
        Zero-copy scatter/gather write with correct partial-write handling.
        """
        # parts will be "consumed" (popped) as they are sent
        while parts:
            # bytes sent
            n_sent = self.sock.sendmsg(parts)
            if n_sent == 0:
                raise ConnectionError("Broken connection during sendmsg")

            sent = 0
            while parts and sent < n_sent:
                p = parts[0]
                p_len = len(p)
                remain = n_sent - sent

                if remain >= p_len:
                    parts.pop(0)
                    sent += p_len
                else:
                    parts[0] = p[remain:]
                    break

    # only support 3 commands
    # GET
    # SET
    # EXISTS

    def make_key_header(self, key: str) -> tuple[memoryview, memoryview]:
        # returns (key_b, key_len_hdr)
        key_b = key.encode()
        key_len_hdr = f"${len(key_b)}\r\n".encode()
        return memoryview(key_b), memoryview(key_len_hdr)

    def get(self, key: str, recv_buf: memoryview):
        """
        assumption:
        both recv_buf and the payload stored in redis for key
        should be of size chunk_size

        recv_buf should be a direct reference to the buffer inside
        of a MemoryObj for zero-copy retrieval
        """
        assert len(recv_buf) == self.chunk_size, "recv_buf is not of size chunk_size"

        key_b, key_len_hdr = self.make_key_header(key)

        # build scatter gather msg
        parts = [
            *self._get_prefix,
            key_len_hdr,
            key_b,
            self.crlf,
        ]

        self._send_multipart(parts)

        # 1. read size header (validation)
        # we could discard the header but validating it is safer
        size_hdr = bytearray(self.size_header_len)
        self._recv_exactly_into(self.size_header_len, memoryview(size_hdr))

        assert size_hdr == self.size_header, "GET command returned invalid size header"

        # 2. read the payload / KV Cache directly into the recv_buf
        self._recv_exactly_into(self.chunk_size, recv_buf)

        # 3. read the trailer (validation)
        # we could discard the trailer but validating it is safer
        trailer = bytearray(self.crlf_len)
        self._recv_exactly_into(self.crlf_len, memoryview(trailer))
        assert trailer == self.crlf, "GET command returned invalid trailer"

    def set(self, key: str, send_buf: memoryview):
        """
        assumption: send_buf is of size chunk_size
        """
        assert len(send_buf) == self.chunk_size, "send_buf is not of size chunk_size"

        key_b, key_len_hdr = self.make_key_header(key)

        # build scatter gather msg
        parts = [
            *self._set_prefix,
            key_len_hdr,
            key_b,
            self.crlf,
            self.size_header,
            send_buf,
            self.crlf,
        ]

        self._send_multipart(parts)

        # expect the ok response
        ret = bytearray(self._ok_len)
        self._recv_exactly_into(self._ok_len, memoryview(ret))
        assert ret == self._ok, "SET command returned invalid response"

    def exists(self, key: str) -> bool:
        """
        check key existence
        """
        key_b, key_len_hdr = self.make_key_header(key)

        parts = [
            *self._exists_prefix,
            key_len_hdr,
            key_b,
            self.crlf,
        ]

        self._send_multipart(parts)

        # read the response
        ret = bytearray(self._int_len)
        self._recv_exactly_into(self._int_len, memoryview(ret))
        if ret == self._one:
            return True
        elif ret == self._zero:
            return False
        else:
            raise ValueError("EXISTS command returned invalid response")

    def _recv_int_response(self) -> int:
        """
        When we don't know beforehand the size of the response
        """
        tmp = bytearray()
        while True:
            # read one byte at a time
            b = self.sock.recv(1)
            if len(tmp) == 0:
                assert b == b":"
            if not b:
                raise ConnectionError("Socket closed while reading")
            tmp.append(b[0])
            if len(tmp) >= 2 and tmp[-2:] == b"\r\n":
                # exclude the prefix : and the CRLF
                return int(tmp[1:-2])

    def batched_exists(self, keys: List[str]) -> int:
        # TODO: buggy because sock.sendmsg has a limit to bytes that can be sent at once
        parts = [*self._exists_prefix] + [
            item
            for key in keys
            for item in (
                self.make_key_header(key)[1],
                self.crlf,
                self.make_key_header(key)[0],
                self.crlf,
            )
        ]
        self._send_multipart(parts)

        return self._recv_int_response()

    def close(self):
        self.sock.close()


class MultiRESPClient:
    """
    Multithreaded wrapper around RESPClient

    Please pass in keys with string serialization
    """

    def __init__(self, host: str, port: int, chunk_size: int, num_threads: int):
        self.num_threads = num_threads
        # i probably does not need to be protected
        # self.dispatch_lock = threading.Lock()
        self.i = 0  # round robin index for the dispatcher

        self.queues: list[queue.Queue] = [queue.Queue() for _ in range(num_threads)]
        self.clients = [RESPClient(host, port, chunk_size) for _ in range(num_threads)]

        self.threads = [
            threading.Thread(
                target=self.worker_loop,
                args=(self.clients[i], self.queues[i]),
                daemon=True,
            )
            for i in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def worker_loop(self, client: RESPClient, q: queue.Queue):
        while True:
            op, key, buf, future = q.get()
            try:
                # opcodes: get, set, exists
                if op == "get":
                    client.get(key, buf)
                    future.set_result(None)

                elif op == "set":
                    client.set(key, buf)
                    future.set_result(None)

                elif op == "exists":
                    exists = client.exists(key)
                    future.set_result(exists)

                elif op == "close":
                    client.close()
                    break  # exit loop

                else:
                    raise ValueError(f"Invalid operation: {op}")
            except Exception as e:
                if future:
                    future.set_exception(e)
            finally:
                q.task_done()

    def _dispatch(self, item):
        """
        Dispatch a job to a worker RESPClient
        """
        # item: (op, key, buf, future)
        i = self.i
        self.i = (i + 1) % self.num_threads
        # the default size is infinite so .put() should never block
        self.queues[i].put(item)

    def set(self, key, buf):
        f = Future()
        self._dispatch(("set", key, buf, f))
        return f

    def get(self, key, buf):
        f = Future()
        self._dispatch(("get", key, buf, f))
        return f

    def exists(self, key):
        f = Future()
        self._dispatch(("exists", key, None, f))
        return f

    def close(self):
        for i in range(self.num_threads):
            self._dispatch(("close", None, None, None))
        for thread in self.threads:
            thread.join()


class RESPConnector(RemoteConnector):
    """
    The remote url should start with "resp://" and only have one host-port pair
    """

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        num_threads: int = 8,
    ):
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # self.full_chunk_size is set in the base class
        # we also get:
        # self.meta_shapes, self.meta_dtypes, self.meta_fmt
        self.host = host
        self.port = port
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        # empirically, num_threads >=4 seems to be around the same
        self.connection = MultiRESPClient(host, port, self.full_chunk_size, num_threads)

        self.pq_executor = AsyncPQExecutor(loop)

    async def _exists(self, key: CacheEngineKey) -> bool:
        f = self.connection.exists(key.to_string())
        return await asyncio.wrap_future(f)

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        f = self.connection.exists(key.to_string())
        return f.result()

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        """
        especially for the RESPConnector,
        the importance of batched operations is to make sure that dispatches
        are done as soon as possible without the possibility of asyncio's
        event loop delaying the scheduling of command dispatch
        """
        key_strs = [key.to_string() for key in keys]
        futures = [self.connection.exists(key_str) for key_str in key_strs]
        results = [future.result() for future in futures]
        return sum(results)

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # TODO: does not account for eviction between exists() and get()
        key_str = key.to_string()
        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
        )

        # byte array view of tensor
        recv_buf = memory_obj.byte_array
        f = self.connection.get(key_str, recv_buf)
        await asyncio.wrap_future(f)

        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def support_batched_get(self) -> bool:
        return True

    async def _batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        for the RESPConnector in particular,
        the importance of batched operations is to make sure that dispatches
        are done as soon as possible without the possibility of asyncio's
        event loop delaying the scheduling of command dispatch
        """
        key_strs = [key.to_string() for key in keys]
        memory_objs = self.local_cpu_backend.batched_allocate(
            self.meta_shapes, self.meta_dtypes, len(keys), self.meta_fmt
        )

        if memory_objs is None or None in memory_objs:
            logger.warning("Failed to allocate memory for some keys")
            return [None] * len(keys)

        recv_bufs = [memory_obj.byte_array for memory_obj in memory_objs]
        futures = [
            asyncio.wrap_future(self.connection.get(key_str, recv_buf))
            for key_str, recv_buf in zip(key_strs, recv_bufs, strict=False)
        ]
        await asyncio.gather(*futures)
        return memory_objs

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        return await self.pq_executor.submit_job(
            self._batched_get, keys=keys, priority=Priorities.GET
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        send_buf = memory_obj.byte_array
        f = self.connection.set(key_str, send_buf)
        await asyncio.wrap_future(f)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        """
        for the RESPConnector in particular,
        the importance of batched operations is to make sure that dispatches
        are done as soon as possible without the possibility of asyncio's
        event loop delaying the scheduling of command dispatch
        """
        key_strs = [key.to_string() for key in keys]
        send_bufs = [memory_obj.byte_array for memory_obj in memory_objs]
        futures = [
            asyncio.wrap_future(self.connection.set(key_str, send_buf))
            for key_str, send_buf in zip(key_strs, send_bufs, strict=False)
        ]
        await asyncio.gather(*futures)

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        self.connection.close()
        logger.info("Closed the RESP connection")

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        key_strs = [key.to_string() for key in keys]
        futures = [
            asyncio.wrap_future(self.connection.exists(key_str)) for key_str in key_strs
        ]

        results = await asyncio.gather(*futures)
        return sum(results)

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """
        same implementation as batched_get
        """
        key_strs = [key.to_string() for key in keys]
        memory_objs = self.local_cpu_backend.batched_allocate(
            self.meta_shapes, self.meta_dtypes, len(keys), self.meta_fmt
        )
        if memory_objs is None or None in memory_objs:
            logger.warning("Failed to allocate memory for some keys")
            return []

        recv_bufs = [memory_obj.byte_array for memory_obj in memory_objs]
        futures = [
            asyncio.wrap_future(self.connection.get(key_str, recv_buf))
            for key_str, recv_buf in zip(key_strs, recv_bufs, strict=False)
        ]
        results = await asyncio.gather(*futures)
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )


class RedisConnector(RemoteConnector):
    """
    The remote url should start with "redis://", "rediss://", or "unix://",
    and only have one host-port pair
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # set a large max
        self.max_connections = 150
        # redis will crash if we have more than max_connections connections
        self.sem = asyncio.Semaphore(self.max_connections)
        self.pool = redis.ConnectionPool.from_url(
            url, max_connections=self.max_connections
        )
        self.connection = redis.Redis.from_pool(self.pool)
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.pq_executor = AsyncPQExecutor(loop)

    async def _exists(self, key: CacheEngineKey) -> bool:
        async with self.sem:
            return bool(await self.connection.exists(key.to_string() + "metadata"))

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(future.result())

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        async with self.sem:
            metadata_bytes = await self.connection.get(key_str + "metadata")

            if metadata_bytes is None:
                return None

            assert not inspect.isawaitable(metadata_bytes)

            metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

            memory_obj = self.local_cpu_backend.allocate(
                metadata.shapes,
                metadata.dtypes,
                metadata.fmt,
            )
            if memory_obj is None:
                logger.warning("Failed to allocate memory during remote receive")
                return None

            # TODO(Jiayi): Find a way to do `get` inplace
            kv_bytes = await self.connection.get(key_str + "kv_bytes")
        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A better way is to aggregate metadata
            # and kv cache in one key.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            async with self.sem:
                await self.connection.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[: metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[: metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[: metadata.length] = converted

        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        # calling self.put will create a circular dependency
        await asyncio.gather(
            *(
                self._put(key, memory_obj)
                for key, memory_obj in zip(keys, memory_objs, strict=False)
            )
        )

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shapes, kv_dtypes, memory_format
        ).serialize()

        key_str = key.to_string()
        # kv bytes needs to be set first to avoid race condition
        async with self.sem:
            await self.connection.set(key_str + "kv_bytes", kv_bytes)
            await self.connection.set(key_str + "metadata", metadata_bytes)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        await self.connection.close()
        logger.info("Closed the redis connection")

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            async with self.sem:
                if not await self.connection.exists(key.to_string() + "metadata"):
                    return num_hit_counts
            num_hit_counts += 1
        return num_hit_counts

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # calling self.get will create a circular dependency
        results = await asyncio.gather(*(self._get(key) for key in keys))
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )


class RedisSentinelConnector(RemoteConnector):
    """
    Uses redis.Sentinel to connect to a Redis cluster.
    The hosts are specified in the config file, started with "redis-sentinel://"
    and separated by commas.

    Example:
        remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"

    Extra environment variables:
    - REDIS_SERVICE_NAME (required) -- service name for redis.
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    ENV_REDIS_TIMEOUT = "REDIS_TIMEOUT"
    ENV_REDIS_SERVICE_NAME = "REDIS_SERVICE_NAME"

    def __init__(
        self,
        hosts_and_ports: List[Tuple[str, int]],
        username: str,
        password: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # Get service name
        match os.environ.get(self.ENV_REDIS_SERVICE_NAME):
            case None:
                logger.warning(
                    f"Environment variable {self.ENV_REDIS_SERVICE_NAME} is "
                    f"not found, using default value 'redismaster'"
                )
                service_name = "redismaster"
            case value:
                service_name = value

        timeout: float = -1000.0

        # Get timeout
        match os.environ.get(self.ENV_REDIS_TIMEOUT):
            case None:
                timeout = 1
            case value:
                timeout = float(value)

        logger.info(f"Host and ports: {hosts_and_ports}")
        self.sentinel = redis.Sentinel(hosts_and_ports, socket_timeout=timeout)
        self.master = self.sentinel.master_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )
        self.slave = self.sentinel.slave_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )

        self.local_cpu_backend = local_cpu_backend

    async def exists(self, key: CacheEngineKey) -> bool:
        return bool(self.slave.exists(key.to_string() + "metadata"))

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return bool(self.slave.exists(key.to_string() + "metadata"))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        metadata_bytes = self.slave.get(key_str + "metadata")

        if metadata_bytes is None:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shapes,
            metadata.dtypes,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # TODO(Jiayi): Find a way to do `get` inplace
        kv_bytes = self.slave.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A background sweeper might be better
            # for the sake of performance.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            self.master.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[0 : metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[0 : metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[0 : metadata.length] = converted

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shapes, kv_dtypes, memory_format
        ).serialize()

        key_str = key.to_string()
        # kv bytes needs to be set first to avoid race condition
        self.master.set(key_str + "kv_bytes", kv_bytes)
        self.master.set(key_str + "metadata", metadata_bytes)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.master.close()
        self.slave.close()


class RedisClusterConnector(RemoteConnector):
    """
    The remote url starts with "redis-cluster:// and can include one or
    multiple hosts:ports, separated by commas.

    Example:
        remote_url: "redis-cluster://host1:7000,host2:7000,host3:7000"

    Extra environment variables:
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    def __init__(
        self,
        hosts_and_ports: List[Tuple[str, int]],
        username: str,
        password: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # Convert hosts_and_ports to startup_nodes format expected by RedisCluster
        startup_nodes = [ClusterNode(h, p) for (h, p) in hosts_and_ports]

        # set a large max
        self.max_connections = 150
        # redis will crash if we have more than max_connections connections
        self.sem = asyncio.Semaphore(self.max_connections)

        # Initialize cluster connection
        self.cluster = RedisCluster(
            startup_nodes=startup_nodes,
            username=username,
            password=password,
            max_connections=self.max_connections,
            decode_responses=False,
        )
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.pq_executor = AsyncPQExecutor(loop)

    async def _exists(self, key: CacheEngineKey) -> bool:
        async with self.sem:
            return bool(await self.cluster.exists(key.to_string() + "metadata"))

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(future.result())

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        async with self.sem:
            metadata_bytes = await self.cluster.get(key_str + "metadata")

            if metadata_bytes is None:
                return None

            assert not inspect.isawaitable(metadata_bytes)

            metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

            memory_obj = self.local_cpu_backend.allocate(
                metadata.shapes,
                metadata.dtypes,
                metadata.fmt,
            )
            if memory_obj is None:
                logger.warning("Failed to allocate memory during remote receive")
                return None

            # TODO(Jiayi): Find a way to do `get` inplace
            kv_bytes = await self.cluster.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A better way is to aggregate metadata
            # and kv cache in one key.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            async with self.sem:
                await self.cluster.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[: metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[: metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[: metadata.length] = converted

        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        # calling self.put will create a circular dependency
        await asyncio.gather(
            *(
                self._put(key, memory_obj)
                for key, memory_obj in zip(keys, memory_objs, strict=False)
            )
        )

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shapes, kv_dtypes, memory_format
        ).serialize()

        key_str = key.to_string()
        # kv bytes needs to be set first to avoid race condition
        async with self.sem:
            await self.cluster.set(key_str + "kv_bytes", kv_bytes)
            await self.cluster.set(key_str + "metadata", metadata_bytes)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        await self.cluster.close()
        logger.info("Closed the redis cluster connection")

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            async with self.sem:
                if not await self.cluster.exists(key.to_string() + "metadata"):
                    return num_hit_counts
            num_hit_counts += 1
        return num_hit_counts

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # calling self.get will create a circular dependency
        results = await asyncio.gather(*(self._get(key) for key in keys))
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )
