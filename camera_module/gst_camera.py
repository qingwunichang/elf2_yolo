import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import threading

Gst.init(None)

class GSTCamera:
    def __init__(self, device="/dev/video21", width=640, height=480):
        self.width = width
        self.height = height
        self.device = device
        self.frame = None
        self.running = True
        if device == "/dev/video21":
            pipeline_str = (
                f"v4l2src device={device} ! "
                f"video/x-raw,width={width},height={height},format=YUY2 ! "
                f"videoconvert ! video/x-raw,format=RGB ! "
                f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
            )
        else:
            pipeline_str = (
                f"v4l2src device={device} ! "
                f"video/x-raw,width={width},height={height},format=NV12 ! "
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
            )

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self._on_new_sample)

        self.mainloop = GLib.MainLoop()
        self.thread = threading.Thread(target=self.mainloop.run, daemon=True)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.thread.start()

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            data = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = data.reshape((self.height, self.width, 3))
            self.frame = frame.copy()
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def read(self):
        return self.frame

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        if self.mainloop.is_running():
            self.mainloop.quit()
        self.thread.join()
