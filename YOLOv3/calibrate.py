import pyrealsense2 as rs
import numpy as np
import cv2
# point = (200, 150)
#
# def show_distance(event, x, y, args, params):
#     global point
#     point = (x, y)
#     # print(1)

if __name__ == "__main__":

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    #设置将深度图对齐到彩色图
    align_to = rs.stream.color
    alignedFs = rs.align(align_to)
    #深度图彩色化工具
    c = rs.colorizer(0)

    profile = pipeline.start(config)

    #因为我们将深度图对齐到了彩色图，所以我们用彩色图的内参
    profile = profile.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()

    mtx = np.array([[intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1]])
    dist = np.array([intr.coeffs])
    print(mtx)
    print(dist)

    print()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = alignedFs.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print('Error')
            exit()

        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        # distance = depth[point[1], point[0]]
        # cv2.setMouseCallback("color", show_distance)
        # cv2.putText(color, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
        #             (0, 0, 0), 2)
        # cv2.setMouseCallback("color", show_distance)


        # print(depth[240,320])
        # color = np.asanyarray(color_frame.get_data())
        # 深度图彩色化
        depth_colormap = np.asanyarray(c.colorize(depth_frame).get_data())

        cv2.imshow("color", color)
        cv2.imshow("depth", depth)
        cv2.imshow("depth_colormap", depth_colormap)
        key = cv2.waitKey(30)
        #按下空格开始标定
        if key == ord(' '):
            try:
                print('开始标定')
                criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

                # 初始化获取标定板角点的位置
                objp = np.zeros((11 * 8, 3))
                # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
                objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

                obj_points = objp * 10  # 这里乘以的值是标定板格子的宽度

                # 图片转为灰度图，找到角点
                # color = cv2.imread('./chessboard/test001.png')
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
                #
                # print("下面是corners：")
                # print(corners)

                # 画图
                for i, idx in zip(corners,range(len(corners))):
                    cv2.putText(color, str(idx),  (int(i[0][0]), int(i[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    cv2.circle(color, (int(i[0][0]), int(i[0][1])), 3, [255, 0, 0], -1)

                # 求解
                ret, rvecs, tvecs = cv2.solvePnP(obj_points, corners, mtx, dist)
                rotation, _ = cv2.Rodrigues(rvecs)



                print("内参矩阵")
                print(mtx)
                print("畸变向量")
                print(dist)
                print("旋转向量")
                print(rvecs)
                print("平移向量")
                print(tvecs)
                print("旋转矩阵")
                print(rotation)
                print(rotation.dot(rotation.T))
                # cv2.namedWindow("color", 0)
                # cv2.resizeWindow("color", 1280, 960)
                # color = cv2.resize(color, (1280, 960))
                cv2.imshow("color", color)
                # cv2.imshow("depth", depth)
                # cv2.imshow("depth_colormap", depth_colormap)
                cv2.waitKey(-1)



            except:
                print("错误请重试")

        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

