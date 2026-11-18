#include "camera/live_pipeline.h"

// camera_live 的薄入口：参数/default split 在 live_cli.cpp，
// 三 Worker、统一轨迹 ID 与线程退出流程在 live_pipeline.cpp。
int main(int argc, char** argv) {
    return yolo::live::runLiveCli(argc, argv);
}
