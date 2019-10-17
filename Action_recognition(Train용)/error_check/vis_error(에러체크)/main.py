import sys
import os
import numpy as np
from vispy import app, scene, io
# from vispy import color

canvas = scene.SceneCanvas(title='Skeleton Pose', size=(600, 600), keys='interactive')
view = canvas.central_widget.add_view(camera='turntable')
view.camera.fov = 45
view.camera.distance = 10

pose_data_fname = './data/error.csv'
data = np.loadtxt(pose_data_fname, skiprows=1, delimiter=',')
pose_data = data[:, :-2]
predict = data[:, -2]
real = data[:, -1]

predict = predict.reshape(-1)
real = real.reshape(-1)
pose_data = pose_data.reshape(-1,19,3)

pose_idx = 0
test = 0
max_pose_idx = pose_data.shape[0] - 1

pos = pose_data[pose_idx]
j_colors = np.random.uniform(size=(19, 3), low=.5, high=.8)

# j_colors = color.get_colormap('orange').map(np.array([e/24 for e in range(25)]))
b_colors = (0.9, 0.6, 0.1, 0.8)
tj_colors = (0.9, 0.9, 0.1, 0.3)
tj_idx = [3, 4, 5, 6, 8, 9, 10]

bones = [[0, 19], [19, 1], [1, 2],
         [19, 3], [3, 4], [4, 5], [5, 6],
         [19, 7], [7, 8], [8, 9], [9, 10],
         [11, 12], [12, 13], [13, 14],
         [15, 16], [16, 17], [17, 18]]

playing_timer = app.Timer()
is_playing = False
playing_dt = 0.033
tj_len = 30

p1 = scene.visuals.Markers()
p1.set_data(pos, face_color=j_colors)
b1 = scene.visuals.Line(pos, color=b_colors, width=5, connect=np.array(bones, dtype=np.uint8))
xyza = scene.visuals.XYZAxis()
text_predict = scene.visuals.Text(str(predict[test]), color='yellow', font_size=64,pos = (.1,.0))
text_real = scene.visuals.Text(str(real[test]), color='yellow', font_size=64,pos = (-.1,.0))
tj1 = [scene.visuals.Line(pos, color=j_colors, width=2) for _ in tj_idx]
show_trailj = False
for j in tj1:
    j.visible = show_trailj

view.add(p1)
view.add(b1)
for j in tj1: view.add(j)
view.add(xyza)
view.add(text_predict)
view.add(text_real)


def set_pose(idx):
    global pose_idx, pose_data, pos, text_pose_idx, test,real,predict
    if idx < 0: idx = 0
    if idx > max_pose_idx: idx = max_pose_idx
    pose_idx = idx
    test = pose_idx // 30
    print(test)
    pos = pose_data[pose_idx]
    p1.set_data(pos, face_color=j_colors)
    b1.set_data(pos)
    text_predict.text = str(predict[test])
    text_real.text = str(real[test])

    for n in range(len(tj_idx)):
        p0 = max(0, pose_idx - tj_len)
        tj1[n].set_data(pose_data[p0:pose_idx, tj_idx[n]], color=tj_colors)


def play_forward(ev):
    global pose_idx, max_pose_idx, is_playing
    if pose_idx < max_pose_idx:
        set_pose(pose_idx + 1)
    elif pose_idx >= max_pose_idx:
        playing_timer.stop()
        is_playing = False


playing_timer.connect(play_forward)
# playing_timer.start(playing_dt)


@canvas.connect
def on_key_press(e):
    # print("{} {}".format(e.text, e.key))
    global pose_idx, is_playing, playing_dt, text_pose_idx, show_trailj
    if e.text == ',':
        set_pose(pose_idx - 1)
    elif e.text == '.':
        set_pose(pose_idx + 1)
    elif e.text == '<':
        set_pose(pose_idx - 10)
    elif e.text == '>':
        set_pose(pose_idx + 10)
    elif e.key == 'Home':
        set_pose(0)
    elif e.key == 'End':
        set_pose(max_pose_idx)
    elif e.key == 'PageDown':
        set_pose(pose_idx - 100)
    elif e.key == 'PageUp':
        set_pose(pose_idx + 100)
    elif e.text == ' ':
        if is_playing:
            playing_timer.stop()
            is_playing = False
        else:
            playing_timer.start(playing_dt)
            is_playing = True
    elif e.text == 'f':
        text_pose_idx.visible = not text_pose_idx.visible
    elif e.text == 't':
        show_trailj = not show_trailj
        for j in tj1: j.visible = show_trailj
    elif e.text == 'P':
        img = canvas.render(bgcolor='white')
        img_fname = '{}{}{}{:08d}{}'.format('snapshot_', os.path.basename(pose_data_fname).split('.')[0], '_', pose_idx, '.png')
        io.write_png(img_fname, img)


if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()