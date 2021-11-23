#ret, image = cv2.imreadmulti('LanB2_KD1P2_31603_gut_2.tif', [], cv2.IMREAD_ANYCOLOR)
# You can then access each layer by their indexes (eg. image[0]).

from matplotlib import pyplot as plt
import time
import pandas as pd
import math
import numpy as np
import cv2
from skimage import io

import scipy.stats

ret, images = cv2.imreadmulti(
    'C:/Users/marci/Documents/projetos_code/bands fish/videos/histo.tif', [], cv2.IMREAD_GRAYSCALE)

'''images_corrected = []
for image in images:
  print("image")
  print(image)
  max_number = 50
  image = image.astype(np.uint16)
  corrected = np.multiply(image, 255) #.astype(np.uint16)
  corrected_divided = np.divide(corrected, max_number).astype(np.uint8)
  #corrected_divided_bright = np.add(corrected_divided, 50).astype(np.uint8)

  print("multipled")
  print(corrected)
  images_corrected.append(corrected_divided)
  print("corrected")
  print(images_corrected[-1])
  print(np.max(images_corrected[-1]))

images = images_corrected
'''

avg_img = np.mean(images, axis=0)
#result = scipy.stats.mode(np.stack(images[::10]), axis=0)
#avg_img = result.mode[0]

avg_img_as_background = avg_img.astype(np.uint8)


imSx = cv2.resize(avg_img_as_background, (1000, 500))               
cv2.imshow("avg_img", imSx)  
  
  

blured_bg_image = cv2.GaussianBlur(avg_img_as_background, (9, 9), 0)

blank_image = np.zeros((blured_bg_image.shape[0], blured_bg_image.shape[1], 3), np.uint8)

final_counts = []

for image in images:
  blured_main_image = cv2.GaussianBlur(image, (9, 9), 0)
  diff = cv2.absdiff(blured_bg_image, blured_main_image)
  ret, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
  #ret2,thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  filtered_contours = []
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  for idx, cnt in enumerate(contours):      
    area = cv2.contourArea(cnt)    
    if area > 15:
        filtered_contours.append(cnt)
      
  for c in filtered_contours:
    # compute the center of the contour
    try:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        cv2.circle(image, (cX, cY), 20, (0, 0, 255), 5)
    
    except:
        pass 
          
          
    #drawn_image = blank_image.copy()
    #drawn_image = cv2.drawContours(drawn_image, contours, -1, color=(0,255,0),thickness=-1)
    #imS = cv2.resize(drawn_image, (1000, 500))               
    #cv2.imshow("result", imS)
    
  imSs = cv2.resize(image, (1000, 500))               
  cv2.imshow("original", imSs)  
  
  if cv2.waitKey(200) & 0xFF == ord('q'):
      break
      


  '''for img in final_img:
    imS = cv2.resize(img, (960, 540))               
    cv2.imshow("output", imS)
    time.sleep(0.5)                   
    '''


  '''
  print("updt")
  print(updated_array_int[0])
  print(updated_array_int[0].shape)
  print(type(updated_array_int[0]))
  # plt.imshow(imagex[0])
  # plt.show()
  '''


# cv2.destroyAllwindows()


#background_image = cv2.imread('C:/Users/marci/Desktop/background_1.jpg')
#bw_back = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
#bw_back = cv2.GaussianBlur(bw_back, (9,9) ,0)

# create a blank image to plot everything on
#blank_image = np.zeros((bw_back.shape[0], bw_back.shape[1], 3), np.uint8)

#previous_df = None
#lopp = 0

#previous_id_fish_local = []

#cv2.imshow('background' , bw_back)

# Check if camera opened successfully
'''if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
for i in range(3850,6000):   #3000 to 4000'''

'''
  cap.set(1, i)
  print(i)
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    cv2.imshow('Main' , frame)

    # Display the resulting frame
    bw_mainImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bw_mainImage = cv2.GaussianBlur(bw_mainImage, (9,9) ,0)

    diff = cv2.absdiff(bw_back, bw_mainImage)
          
    ret,thresh = cv2.threshold(diff,15,255,cv2.THRESH_BINARY) 


  
    

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    lenght_of_fish_local = []
    position_fish_local = []
    quadrant_local = []
    fish_tail_local = []
    fish_head_local = []
    idx_local = [] 
    id_fish_local = []
    quadrant_local = []
    fish_area = []
    #fish_color = []

    counter = 0
    for idx, cnt in enumerate(contours):      
        area = cv2.contourArea(cnt)        


        if area > 100 and area < 1500:          

          #will be used to predict the size of the fish excluding the tail part
          fish_total_pixels = len(cnt)

          area = cv2.contourArea(cnt)


          #to define in which quadrant the fish belongs to
          M = cv2.moments(cnt)
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"]) 

          # determine the most extreme points along the contour
          extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
          extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
          extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
          extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
          list_of_points = [extLeft, extRight, extTop, extBot]

          #calculate the tail by discovering the fartherst point from center of mass of fish
          distances_tail = []  
          for l in list_of_points:                     
            distance = math.sqrt(   (cX-l[0])**2 + (cY-l[1])**2    )
            distances_tail.append(distance)

          max_value = max(distances_tail)
          max_index_tail = distances_tail.index(max_value)

          #discovering the head by calculating the fartherst point from tail
          distances_head = []
          for l in cnt:
                   
            distance = math.sqrt(   (l[0][0]-list_of_points[max_index_tail][0])**2 + (l[0][1]-list_of_points[max_index_tail][1])**2    )
            distances_head.append(distance)

          max_value = max(distances_head)
          max_index_head = distances_head.index(max_value)

          arr = np.array(distances_head)
          arr = arr.argsort()[-10:][::-1]
          list_final = arr.tolist()
     

          farthest_values = []
          for d in list_final:
            list_value = cnt[d].tolist()              
            farthest_values.append(list_value)

          arr = np.array(farthest_values)
          aver_head = np.mean(arr, axis=0).astype(int)
          aver_head = (aver_head[0][0], aver_head[0][1])


          #calculate the center of mass of the fish excluding the tail
          distances_cm = []
          for j in cnt:
                   
            #append distances betwen head and each point of the fish
            distance = math.sqrt(   (j[0][0]-int(aver_head[0]))**2 + (j[0][1]-int(aver_head[1]))**2    )
            distances_cm.append(distance)            

          slowest_indices = sorted(range(len(distances_cm)), key = lambda sub: distances_cm[sub])[:int(fish_total_pixels*0.8)]                   
          
          nearests_values = []
          for e in slowest_indices:
            list_value_cm = cnt[e].tolist()              
            nearests_values.append(list_value_cm)

          arr = np.array(nearests_values)
          aver_cm = np.mean(arr, axis=0).astype(int)        

          fish_pectoral_lenght = math.sqrt( (aver_head[0] - aver_cm[0][0])  **2 + (aver_head[1] - aver_cm[0][1])**2    )

        
          if (aver_cm[0][0] < 427 and aver_cm[0][1] < 417):  # belongs to quadrant A1
            quadrant_value = "A"
          elif (aver_cm[0][0] < 427 and aver_cm[0][1] > 417):
            quadrant_value = "B"
          elif (aver_cm[0][0] > 427 and aver_cm[0][1] < 417):
            quadrant_value = "C"
          else:
            quadrant_value = "D"       



          #store variables locally in the loop for imediate calculation purposes
          idx_local.append(counter)          
          lenght_of_fish_local.append(fish_pectoral_lenght)
          position_fish_local.append((aver_cm[0][0], aver_cm[0][1]))
          fish_tail_local.append(list_of_points[max_index_tail])
          fish_head_local.append(aver_head)
          quadrant_local.append(quadrant_value)
          fish_area.append(area)
          #fish_color.append(color)


          counter +=1

    
    dframe = pd.DataFrame(idx_local)
    dframe['lenght_of_fish_local'] = lenght_of_fish_local
    dframe['position_fish_local'] = position_fish_local
    dframe['fish_tail_local'] = fish_tail_local
    dframe['fish_head_local'] = fish_head_local
    dframe['quadrant_local'] = quadrant_local
    dframe['fish_area'] = fish_area
    dframe["fish_id"] = np.nan
    #dframe['fish_color'] = np.nan

    if previous_df is None:
      
      previous_df = dframe.copy()

      print(previous_df)
      #previous_df['fish_id'] = range(1, 1+len(previous_df))
      previous_df.loc[previous_df['quadrant_local'] == quadr, 'fish_id'] = [x for x in [1, 2]]
    
  
    unique_quadrants = dframe.quadrant_local.unique()
    for index_q, row_q in enumerate(unique_quadrants):      
      fish_per_quadrant = (dframe['quadrant_local']==quadr).sum()

      if row_q == quadr:

        print(update_counter)    

        if fish_per_quadrant < 2:        
          update_counter = 0
          if counter_activation == 1:
            dframe_last_seen = previous_df.copy()
            counter_activation = 0         
          continue

        if (update_counter == 0 or update_counter == 1) and fish_per_quadrant == 2:           
            dframe.loc[dframe['quadrant_local'] == quadr, 'fish_id'] = [x for x in ['X', 'Y']]
            previous_df = dframe.copy()
            
            continue      
     
       
        filtered_df = dframe[(dframe.quadrant_local == row_q)]
        filtered_df.index = filtered_df.index.set_names(['original_index'])
        filtered_df = filtered_df.reset_index()        
          
      
        
             
        for idx, row in filtered_df.iterrows():
          distances_indices = [] 
          current_position_fish_local = row['position_fish_local']          
                      

          if update_counter == 90 or update_counter == 91:

            previous_fish_1 = dframe_last_seen.loc[dframe_last_seen['quadrant_local'] == row_q].iloc[0]   
            previous_fish_2 = dframe_last_seen.loc[dframe_last_seen['quadrant_local'] == row_q].iloc[1]

            similarity_1 = abs(row['fish_area'] - previous_fish_1['fish_area'])
            similarity_2 = abs(row['fish_area'] - previous_fish_2['fish_area'])

            if similarity_1 < similarity_2:

              dframe.loc[row['original_index'],'fish_id'] = int(previous_fish_1['fish_id']) 
            else:
              dframe.loc[row['original_index'],'fish_id'] = int(previous_fish_2['fish_id'])

            activate_counter = 1

            counter_activation = 1

            continue
     
       

          previous_fish_1 = previous_df.loc[previous_df['quadrant_local'] == row_q].iloc[0]   
          previous_fish_2 = previous_df.loc[previous_df['quadrant_local'] == row_q].iloc[1]  
      
          previous_fish_1_id = previous_fish_1['fish_id']
          previous_fish_2_id = previous_fish_2['fish_id']

          previous_fish_1_position = previous_fish_1['position_fish_local']
          previous_fish_2_position = previous_fish_2['position_fish_local']

          previous_fish_1_area = previous_fish_1['fish_area']
          previous_fish_2_area = previous_fish_2['fish_area'] 
          
          distance_value_1 = math.sqrt(   (previous_fish_1_position[0]-current_position_fish_local[0])**2 + (previous_fish_1_position[1]-current_position_fish_local[1])**2    )
          distances_indices.append(distance_value_1)
          distance_value_2 = math.sqrt(   (previous_fish_2_position[0]-current_position_fish_local[0])**2 + (previous_fish_2_position[1]-current_position_fish_local[1])**2    )
          
          distances_indices.append(distance_value_2)
          distances_indices.append(distance_value_1)              
                  
          lower_indice = sorted(range(len(distances_indices)), key = lambda sub: distances_indices[sub])[:1]
      
          if lower_indice[0] == 0:      
            dframe.loc[row['original_index'],'fish_id'] = previous_fish_1_id
            dframe.loc[row['original_index'],'fish_area'] = (previous_fish_1_area * 90 + row['fish_area'])/91
            
            
          else:   
            dframe.loc[row['original_index'],'fish_id'] = previous_fish_2_id
            dframe.loc[row['original_index'],'fish_area'] = (previous_fish_2_area * 90 + row['fish_area'])/91
           

          
    update_counter += 1
    if update_counter > 100:
      update_counter = 100

    
    drawn_image = blank_image.copy()
    drawn_image = cv2.drawContours(drawn_image, contours, -1, color=(255,0,0),thickness=-1)


      
    for c in idx_local[:8]:
      #tail plot
      cv2.circle(drawn_image, fish_tail_local[c], 2, (0, 0, 255), -1)

      #head
      cv2.circle(drawn_image, fish_head_local[c], 2, (0, 255, 0), -1)

      #center of mass
      cv2.circle(drawn_image, position_fish_local[c], 2, (0, 165, 255), -1)
      

    position_list = dframe.position_fish_local.tolist()
    fish_id = dframe.fish_id.tolist()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for indice, value in enumerate(fish_id):
      #if math.isnan(value):
        #pass
      #else:
      cv2.putText(drawn_image,str(fish_id[indice]),(position_list[indice]), font, 1,(255,255,255),2)
    

         
    cv2.line(drawn_image, (427, 0), (427,870), (255, 255, 255), thickness=1)
    cv2.line(drawn_image, (0, 417), (870,417), (255, 255, 255), thickness=1)   

    #print(final_image.shape)

    #show the image with filtered countours plotted   
    cv2.imshow('Frame',drawn_image)



    #if fish_per_quadrant == 2:
    previous_df = dframe.copy()
    #else:
      #dframe_last_seen = previous_df.copy()
    

    #usefull if extract one image for background needed
    #if i == 3200:
        #cv2.imwrite('C:/Users/marci/Desktop/background_3.jpg', frame)
  
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break



# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroy'''
