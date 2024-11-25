import os
import multiprocessing
# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

def detect_pcd(file_path):
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)
    # Voxel Downsampling 수행
    voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    # RANSAC을 사용하여 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=2000)

    # 도로에 속하지 않는 포인트 (outliers) 추출
    final_point = ror_pcd.select_by_index(inliers, invert=True)


    # final_point의 점 개수 확인
    num_points = np.asarray(final_point.points).shape[0]

    print(f"final_point에 속하는 점들의 개수는 {num_points}개입니다.")


    # MeanShift 클러스터링
    points = np.asarray(final_point.points)


    mean_shift = MeanShift(bandwidth = 0.7).fit(points)  # bandwidth는 밀도 추정의 크기
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_ 


    # 각 클러스터에서의 이상치 제거 함수
    def remove_outliers_from_clusters(points, labels, cluster_centers, threshold=2.0):
        inlier_points = []  # 이상치가 아닌 점들
        inlier_labels = []  # 이상치가 아닌 클러스터 레이블
        for i in range(np.max(labels) + 1):
            # 각 클러스터의 포인트들
            cluster_points = points[labels == i]
            
            # 클러스터 중심
            cluster_center = cluster_centers[i]
            
            # 각 점과 중심 간의 거리 계산
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            count_above_threshold = np.sum(distances > threshold)
            if count_above_threshold <= len(distances)/10:
                # 지정된 threshold 거리 이하의 점들만 inlier로 간주
                inlier_cluster_points = cluster_points[distances <= threshold]
            else:
                inlier_cluster_points = cluster_points


            # inlier 점들 및 레이블 추가
            inlier_points.append(inlier_cluster_points)
            inlier_labels.append(np.full(inlier_cluster_points.shape[0], i))
            
                
        # 모든 클러스터의 inlier 포인트들을 하나로 합침
        inlier_points = np.vstack(inlier_points)
        inlier_labels = np.concatenate(inlier_labels)
        return inlier_points, inlier_labels

    # 각 클러스터에서 이상치를 제거한 점들
    inlier_points, inlier_labels = remove_outliers_from_clusters(points, labels, cluster_centers, threshold=1.95)


    # 새로운 포인트 클라우드 객체 생성
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(inlier_points)

    # 색상 설정
    max_label = inlier_labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # 노이즈를 제거하고 각 클러스터에 색상 지정
    colors = plt.get_cmap("cividis")(inlier_labels / (max_label + 1 if max_label > 0 else 1))
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 필터링 기준 설정
    # 필터링 기준 1. 클러스터 내 최대 최소 포인트 수
    min_points_in_cluster = 10   # 클러스터 내 최소 포인트 수
    max_points_in_cluster = 45  # 클러스터 내 최대 포인트 수

    # 필터링 기준 2. 클러스터 내 최소 최대 Z값
    min_z_value = -1.0    # 클러스터 내 최소 Z값
    max_z_value = 2.5   # 클러스터 내 최대 Z값

    # 필터링 기준 3. 클러스터 내 최소 최대 Z값 차이
    min_height = 1.0   # Z값 차이의 최소값
    max_height = 1.95   # Z값 차이의 최대값

    max_distance = 40.0  # 원점으로부터의 최대 거리

    max_dist_diff = 1.5
    # 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
    bboxes_1234 = []

    for i in range(np.max(inlier_labels) + 1):
        cluster_indices = np.where(inlier_labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = filtered_pcd.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            x_values = points[:, 0]
            y_values = points[:, 1]
            z_min = z_values.min()
            z_max = z_values.max()
            x_min = x_values.min()
            x_max = x_values.max()
            y_min = y_values.min()
            y_max = y_values.max()
            dist_diff = max((x_max-x_min,y_max-y_min))        
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    
                    if dist_diff <= max_dist_diff:
                        if distances.max() <= max_distance:
                            bbox = cluster_pcd.get_axis_aligned_bounding_box()
                            bbox.color = (1, 0, 0) 
                            bboxes_1234.append(bbox)

    return filtered_pcd, bboxes_1234

# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes,image_path, point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)

    # ViewControl 객체 얻기
    view_control = vis.get_view_control()

    # 카메라 위치를 (0, 0, 1)로 설정
    view_control.set_lookat([0, 20, 0])  # 바라보는 점을 (0, 5, 0)으로 설정
    view_control.set_up([0, 1, 4])      # 카메라의 "위" 방향을 (0, 0, 1)로 설정
    view_control.set_front([0, -4, 3])  # 카메라의 "앞" 방향을 (0, -1, 0)으로 설정
    view_control.set_zoom(0.2)          # 줌을 적절하게 설정 (필요시 조정)



    vis.get_render_option().point_size = point_size
#    vis.run()
#    vis.destroy_window()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(image_path)
    vis.destroy_window()

# pcd 파일 불러오기, 필요에 맞게 경로 수정
directory_path = "data/02_straight_duck_walk/pcd"
output_dir = "output_images_02"
pcd_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pcd')]

for pcd_file in pcd_files:
    filtered_pcd, bboxes_1234 = detect_pcd(pcd_file)

    output_image_path = os.path.join(output_dir, os.path.basename(pcd_file).replace('.pcd', '.png'))
    # 시각화 (포인트 크기를 원하는 크기로 조절 가능)

    visualize_with_bounding_boxes(filtered_pcd, bboxes_1234, image_path=output_image_path, point_size=2.0)


directory_path = "data/03_straight_crawl/pcd"
output_dir = "output_images_03"
pcd_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pcd')]



for pcd_file in pcd_files:
    filtered_pcd, bboxes_1234 = detect_pcd(pcd_file)

    output_image_path = os.path.join(output_dir, os.path.basename(pcd_file).replace('.pcd', '.png'))
    # 시각화 (포인트 크기를 원하는 크기로 조절 가능)

    visualize_with_bounding_boxes(filtered_pcd, bboxes_1234, image_path=output_image_path, point_size=2.0)

directory_path = "data/04_zigzag_walk/pcd"
output_dir = "output_images_04"
pcd_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pcd')]



for pcd_file in pcd_files:
    filtered_pcd, bboxes_1234 = detect_pcd(pcd_file)

    output_image_path = os.path.join(output_dir, os.path.basename(pcd_file).replace('.pcd', '.png'))
    # 시각화 (포인트 크기를 원하는 크기로 조절 가능)

    visualize_with_bounding_boxes(filtered_pcd, bboxes_1234, image_path=output_image_path, point_size=2.0)


directory_path = "data/05_straight_duck_walk/pcd"
output_dir = "output_images_05"
pcd_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pcd')]



for pcd_file in pcd_files:
    filtered_pcd, bboxes_1234 = detect_pcd(pcd_file)

    output_image_path = os.path.join(output_dir, os.path.basename(pcd_file).replace('.pcd', '.png'))
    # 시각화 (포인트 크기를 원하는 크기로 조절 가능)

    visualize_with_bounding_boxes(filtered_pcd, bboxes_1234, image_path=output_image_path, point_size=2.0)


directory_path = "data/06_straight_crawl/pcd"
output_dir = "output_images_06"
pcd_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pcd')]



for pcd_file in pcd_files:
    filtered_pcd, bboxes_1234 = detect_pcd(pcd_file)

    output_image_path = os.path.join(output_dir, os.path.basename(pcd_file).replace('.pcd', '.png'))
    # 시각화 (포인트 크기를 원하는 크기로 조절 가능)

    visualize_with_bounding_boxes(filtered_pcd, bboxes_1234, image_path=output_image_path, point_size=2.0)


directory_path = "data/07_straight_walk/pcd"
output_dir = "output_images_07"
pcd_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pcd')]



for pcd_file in pcd_files:
    filtered_pcd, bboxes_1234 = detect_pcd(pcd_file)

    output_image_path = os.path.join(output_dir, os.path.basename(pcd_file).replace('.pcd', '.png'))
    # 시각화 (포인트 크기를 원하는 크기로 조절 가능)

    visualize_with_bounding_boxes(filtered_pcd, bboxes_1234, image_path=output_image_path, point_size=2.0)
