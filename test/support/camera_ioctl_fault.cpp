// 仅链接进测试入口，不进入生产程序。验证不支持 DMABUF 时明确失败、不回退。
#include <cerrno>
#include <cstdarg>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
extern "C" int __real_ioctl(int, unsigned long, ...);
extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
    va_list values; va_start(values, request);
    void* arg = va_arg(values, void*);
    va_end(values);
    if (request == VIDIOC_REQBUFS && static_cast<v4l2_requestbuffers*>(arg)->memory == V4L2_MEMORY_DMABUF) {
        errno = EINVAL;
        return -1;
    }
    return __real_ioctl(fd, request, arg);
}
