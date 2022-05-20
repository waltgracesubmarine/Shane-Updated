#pragma once

#include <memory>

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QThread>
#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/ui/ui.h"

struct FramePair {
  uint32_t frame_id;
  VisionBuf* frame;
  bool valid = false;
};
const int FRAME_BUFFER_SIZE = 5;
static_assert(FRAME_BUFFER_SIZE <= YUV_BUFFER_COUNT);

class CameraViewWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit CameraViewWidget(std::string stream_name, VisionStreamType stream_type, bool zoom, QWidget* parent = nullptr);
  ~CameraViewWidget();
  void setStreamType(VisionStreamType type) { stream_type = type; }
  void setBackgroundColor(const QColor &color) { bg = color; }
  void setFrameId(uint32_t frame_id) {
    draw_frame_id_updated = frame_id != draw_frame_id;
    draw_frame_id = frame_id;
//    uint32_t frame_jump = std::abs((int)frame_id - (int)prev_frame_id);
    if (frame_id == prev_frame_id ) {
//      draw_frame_id += 1;
      frame_offset += 1;
    } else if (std::abs((int)frame_id - (int)_latest_frame_id) < FRAME_BUFFER_SIZE) {
      frame_offset = std::max((int)frame_offset - std::abs((int)frame_id - (int)prev_frame_id), 0);
      frame_offset = std::min((int)frame_offset, FRAME_BUFFER_SIZE);
    }
//    if (frame_jump > FRAME_BUFFER_SIZE) {
//      frame_offset = 0;
//    }
    prev_frame_id = frame_id;
  }

signals:
  void clicked();
  void vipcThreadConnected(VisionIpcClient *);
  void vipcThreadFrameReceived(VisionBuf *, quint32);

protected:
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override { updateFrameMat(w, h); }
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
  virtual void updateFrameMat(int w, int h);
  void vipcThread();

  bool zoomed_view;
  GLuint frame_vao, frame_vbo, frame_ibo;
  GLuint textures[3];
  mat4 frame_mat;
  std::unique_ptr<QOpenGLShaderProgram> program;
  QColor bg = QColor("#000000");

  std::string stream_name;
  int stream_width = 0;
  int stream_height = 0;
  std::atomic<VisionStreamType> stream_type;
  QThread *vipc_thread = nullptr;

  std::deque<std::pair<uint32_t, VisionBuf*>> frames;
  uint32_t draw_frame_id = 0;
  uint32_t prev_frame_id = 0;
  uint32_t _latest_frame_id = 0;
  uint32_t prev_drawn_frame_id = 0;
  uint32_t frame_offset = 0;
  bool draw_frame_id_updated = false;
  FramePair frame_array[FRAME_BUFFER_SIZE];

protected slots:
  void vipcConnected(VisionIpcClient *vipc_client);
  void vipcFrameReceived(VisionBuf *vipc_client, uint32_t frame_id);
};
