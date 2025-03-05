import ctypes
import fcntl

from pixutils.ioctl import IOW

DMA_BUF_SYNC_READ = (1 << 0)
DMA_BUF_SYNC_WRITE = (2 << 0)
DMA_BUF_SYNC_RW = (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
DMA_BUF_SYNC_START = (0 << 2)
DMA_BUF_SYNC_END = (1 << 2)

# pylint: disable=invalid-name
class struct_dma_buf_sync(ctypes.Structure):
    __slots__ = ['flags']
    _fields_ = [('flags', ctypes.c_uint64)]

DMA_BUF_BASE = 'b'
DMA_BUF_IOCTL_SYNC = IOW(DMA_BUF_BASE, 0, struct_dma_buf_sync)

def dmabuf_sync_start(fd, write = False):
   req = struct_dma_buf_sync()
   req.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ
   if write:
      req.flags |= DMA_BUF_SYNC_WRITE
   fcntl.ioctl(fd, DMA_BUF_IOCTL_SYNC, req, True)

def dmabuf_sync_end(fd, write = False):
   req = struct_dma_buf_sync()
   req.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ
   if write:
      req.flags |= DMA_BUF_SYNC_WRITE
   fcntl.ioctl(fd, DMA_BUF_IOCTL_SYNC, req, True)
