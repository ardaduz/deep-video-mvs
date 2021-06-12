import numpy as np
from path import Path

from dvmvs.keyframe_buffer import KeyframeBuffer, SimpleBuffer


def simulate_keyframe_buffer(test_dataset_path, output_folder, n_measurement_frames):
    test_dataset_path = Path(test_dataset_path)
    scene_folders = sorted(test_dataset_path.listdir())

    test_keyframe_buffer_size = 30
    test_keyframe_pose_distance = 0.1
    test_optimal_t_measure = 0.15
    test_optimal_R_measure = 0.0

    for scene_index in range(0, len(scene_folders)):
        scene_folder = scene_folders[scene_index]
        scene = scene_folder.split("/")[-1]
        print("Simulating scene:", scene, " - ", scene_index, "/", len(scene_folders))

        keyframe_buffer = KeyframeBuffer(buffer_size=test_keyframe_buffer_size,
                                         keyframe_pose_distance=test_keyframe_pose_distance,
                                         optimal_t_score=test_optimal_t_measure,
                                         optimal_R_score=test_optimal_R_measure,
                                         store_return_indices=True)

        poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
        image_filenames = sorted((scene_folder / 'images').files("*.png"))

        output_lines = []
        for i in range(0, len(poses)):
            reference_pose = poses[i]

            # POLL THE KEYFRAME BUFFER
            response = keyframe_buffer.try_new_keyframe(reference_pose, None, index=i)
            if response == 3:
                output_lines.append("TRACKING LOST")
            elif response == 1:
                measurement_frames = keyframe_buffer.get_best_measurement_frames(n_measurement_frames)

                output_line = image_filenames[i].split("/")[-1]
                for (measurement_pose, measurement_image, measurement_index) in measurement_frames:
                    output_line += (" " + image_filenames[measurement_index].split("/")[-1])

                output_line = output_line.strip(" ")
                output_lines.append(output_line)

        output_lines = np.array(output_lines)

        dataset_name = test_dataset_path.split("/")[-1]
        np.savetxt('{}/keyframe+{}+{}+nmeas+{}'.format(output_folder, dataset_name, scene, n_measurement_frames), output_lines, fmt='%s')


def simulate_simple_buffer(test_dataset_path, output_folder, n_skip, n_measurement_frames):
    test_dataset_path = Path(test_dataset_path)
    scene_folders = sorted(test_dataset_path.listdir())

    for scene_index in range(0, len(scene_folders)):
        scene_folder = scene_folders[scene_index]
        scene = scene_folder.split("/")[-1]
        print("Simulating scene:", scene, " - ", scene_index, "/", len(scene_folders))

        simple_buffer = SimpleBuffer(n_measurement_frames, store_return_indices=True)

        poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
        image_filenames = sorted((scene_folder / 'images').files("*.png"))

        output_lines = []
        i = 0
        while i < len(poses):
            reference_pose = poses[i]

            # POLL THE KEYFRAME BUFFER
            response = simple_buffer.try_new_keyframe(reference_pose, None, index=i)
            if response == 0:
                i += n_skip
                continue
            elif response == 2:
                output_lines.append("TRACKING LOST")
                i += 1
                continue
            elif response == 3 or response == 4:
                i += 1
            else:
                measurement_frames = simple_buffer.get_measurement_frames()

                output_line = image_filenames[i].split("/")[-1]
                for (measurement_pose, measurement_image, measurement_index) in measurement_frames:
                    output_line += (" " + image_filenames[measurement_index].split("/")[-1])

                output_line = output_line.strip(" ")
                output_lines.append(output_line)
                i += n_skip

        output_lines = np.array(output_lines)

        dataset_name = test_dataset_path.split("/")[-1]
        n_skip_str = str(n_skip)
        np.savetxt('{}/simple{}+{}+{}+nmeas+{}'.format(output_folder, n_skip_str, dataset_name, scene, n_measurement_frames), output_lines,
                   fmt='%s')


def main():
    output_folder = "../sample-data/indices"
    test_dataset_path = "../sample-data/hololens-dataset"
    simulate_keyframe_buffer(test_dataset_path, output_folder, n_measurement_frames=1)
    simulate_keyframe_buffer(test_dataset_path, output_folder, n_measurement_frames=2)
    simulate_keyframe_buffer(test_dataset_path, output_folder, n_measurement_frames=3)

    # for evaluation of simple selection (comment out the rest if only our keyframe selection method is desired)
    simulate_simple_buffer(test_dataset_path, output_folder, n_skip=10, n_measurement_frames=2)
    simulate_simple_buffer(test_dataset_path, output_folder, n_skip=20, n_measurement_frames=2)


main()
