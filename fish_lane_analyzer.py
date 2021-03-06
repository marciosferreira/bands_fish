import cv2
import pathlib
import os


# the 3 lines bellow you need to configure as your needs
# the path of the images stack. Must end with .tif
path_for_images = 'C:/Users/marci/Documents/projetos_code/bands fish/videos/20211009_cab_exp_1_MMStack_Default.ome.tif'
# how many frames to quit when analyzing.
# Using every frame can increase too much the variability of the data. We suggest at least 3.
frame_space = 3
# can make adjustment if the position of the camera changes
list_grid = [0, 138, 261, 390, 519, 636, 762, 897,
             1008, 1140, 1263, 1383, 1515, 1638, 1761, 1900]


def normVideo(frames):

    norm_frames = []

    #max_in_frames = np.max(frames)

    read_frames = []
    for i in range(0, len(frames), frame_space):
        read_frames.append(i)

        frame = frames[i]

        if (frame is not None) and (frame.size > 0):

            image_enhanced = cv2.equalizeHist(frame)
            image_enhanced = cv2.bilateralFilter(image_enhanced, 9, 100, 100)

        norm_frames.append(image_enhanced)

    return norm_frames, read_frames
#


backSub = cv2.createBackgroundSubtractorMOG2(
    history=50, varThreshold=20, detectShadows=True)
#backSub = cv2.createBackgroundSubtractorKNN(history = 10, dist2Threshold = 800.0, detectShadows = False)
ret, images = cv2.imreadmulti(path_for_images, [], cv2.IMREAD_GRAYSCALE)

final_path = pathlib.PurePath(path_for_images)
file_name = final_path.name

if os.path.exists('results/' + file_name + '.csv'):
    os.remove('results/' + file_name + '.csv')
    print("CSV file exist, I will remove it before creating a new one")
else:
    print("CSV file does not exist, I will crate it")


with open("results/" + file_name + '.csv', 'a') as fd:
    fd.write('frame_number, fish_1, fish_2,fish_3,fish_4,fish_5,fish_6,fish_7,fish_8,fish_9,fish_10,fish_11,fish_12,fish_13,fish_14,fish_15\n')

# normalize the frames
images_norm, read_frames = normVideo(images)


fish = {"fish_1": None, "fish_2": None, "fish_3": None, "fish_4": None, "fish_5": None, "fish_6": None, "fish_7": None,
        "fish_8": None, "fish_9": None, "fish_10": None, "fish_11": None, "fish_12": None, "fish_13": None, "fish_14": None, "fish_15": None}

video_final = []

for idxf, image in enumerate(images_norm):

    fgMask = backSub.apply(image)

    if idxf > 4:

        provisional_fish = {"fish_1": None, "fish_2": None, "fish_3": None, "fish_4": None, "fish_5": None, "fish_6": None, "fish_7": None,
                            "fish_8": None, "fish_9": None, "fish_10": None, "fish_11": None, "fish_12": None, "fish_13": None, "fish_14": None, "fish_15": None}

        img_3channels = cv2.merge((image, image, image))

        filtered_contours = []

        image = cv2.GaussianBlur(image, (13, 13), 0)

        contours, hierarchy = cv2.findContours(
            fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 10:
                filtered_contours.append(cnt)

        for c in filtered_contours:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            for idx, z in enumerate(list_grid):
                if idx < len(list_grid) - 1:
                    if list_grid[idx] < cY < list_grid[idx + 1]:
                        if provisional_fish['fish_' + str(idx + 1)] is None:
                            provisional_fish.update(
                                {'fish_' + str(idx + 1): [(cX, cY)]})
                        else:
                            provisional_fish['fish_' +
                                             str(idx + 1)].append((cX, cY))

        invalids = []
        left = []
        left_values = []
        right = []
        right_values = []
        for key, value in provisional_fish.items():
            if value is not None:

                # decide if right or left based on fish dictionary (fish os previous)
                if fish[key] is not None:
                    if value[0][0] < (fish[key][0] - 5):
                        left.append(key)
                        result_left = int(abs(value[0][0] - fish[key][0]))
                        if result_left > 40:
                            result_left = 40
                        left_values.append(result_left)
                    elif value[0][0] > (fish[key][0] + 5
                                        ):
                        right.append(key)
                        result_right = int(abs(value[0][0] - fish[key][0]))
                        if result_right > 40:
                            result_right = 40
                        right_values.append(result_right)

                if len(value) > 1 and fish[key] is not None:
                    results_distance = []

                    for coords in value:
                        cX_diff = abs(coords[0] - fish[key][0])
                        results_distance.append(cX_diff)

                    min_value = min(results_distance)
                    min_index = results_distance.index(min_value)
                    fish.update({key: value[min_index]})
                else:
                    fish.update({key: value[0]})

            else:
                invalids.append(key)

        final_row = [read_frames[idxf]]
        for item in fish.items():
            if item[1] is not None:
                if item[0] not in invalids:
                    if item[0] in left:
                        value_index = left.index(item[0])
                        cv2.circle(
                            img_3channels, (item[1][0], item[1][1]), left_values[value_index], (255, 0, 0), 5)
                        final_row.append(-1 * int(left_values[value_index]))
                    elif item[0] in right:
                        value_index = right.index(item[0])
                        cv2.circle(
                            img_3channels, (item[1][0], item[1][1]), right_values[value_index], (0, 255, 0), 5)
                        final_row.append(int(right_values[value_index]))
                    else:
                        cv2.circle(img_3channels,
                                   (item[1][0], item[1][1]), 5, (0, 0, 255), 5)
                        final_row.append(0)
                else:
                    cv2.circle(img_3channels,
                               (item[1][0], item[1][1]), 5, (0, 0, 255), 5)
                    final_row.append(0)
            else:
                final_row.append(0)

        with open('results/' + file_name + '.csv', 'a') as fd:
            my_str = ','.join(str(x) for x in final_row)
            my_str = my_str + '\n'
            fd.write(my_str)

        img_w = img_3channels[0].shape[0]
        for coord in list_grid:
            start_point = (0, coord)
            end_point = (img_w, coord)
            img_3channels = cv2.line(
                img_3channels, start_point, end_point, (255, 170, 0), 4)

        imS = cv2.resize(img_3channels, (1000, 500))
        cv2.imshow("output", imS)

        imSs = cv2.resize(fgMask, (1000, 500))
        cv2.imshow("result", imSs)

        video_final.append(imS)

        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height, width, layers = video_final[0].shape
size = (width, height)
out_vid = "C:/Users/marci/Documents/projetos_code/bands fish/videos/output.mp4"
out = cv2.VideoWriter(out_vid, fourcc, 10, size)
for i in video_final:
    out.write(i)
out.release()
