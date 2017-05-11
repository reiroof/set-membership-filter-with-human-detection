#!BPY
import bpy
import os
import time
import math
import numpy as np

# If you want to run this simulation with more than 4 cameras,
# please add cameras by yourself and change their name as follows: Camera5, Camera6, ... CameraN.

D = bpy.data
scene = bpy.context.scene
a = np.array([[1,0,0], [0,0,-1],[0,1,0]])
b = np.array([[1/200,0,0],[0,1/200,0],[0,0,1/200]])
c = a.dot(b)
frame_start = D.scenes["Scene"].frame_start
frame_end = D.scenes["Scene"].frame_end
# Set the number of step
frame_end = 301+1
datapath = "C:/Users/nakamura/Documents/MATLAB/Blender_MATLAB/data"
output_path = "/My Documents/MATLAB/Blender_MATLAB"

# Please input the direction of box and human.
# (Which direction you want to move the box and human
#  If you want to move the box to -y direction,
#  then please input [0, -1, 0].)


# scene.render.resolution_percentage = 0
# bpy.context.scene.frame_current = frame_start
# bpy.ops.render.render()
# start_human = array(D.objects["slow_walk_path"].pose.bones["Hips"].location)
# start_box = abs(box_dir) * start_human
# D.objects["box"].location = start_box - 0.2 * box_dir


g = open(datapath + "/param.txt")
h = open(datapath + "/posi_lam.txt")
p = open(datapath"/camera.txt")
#l = open(datapath + "ell_rot.txt")

param = g.readlines()
posi_lam = h.readlines()
camera = p.readlines()
g.close()
h.close()
p.close()

# Set the image size
scene.render.resolution_x = int(param[0].split()[1])
scene.render.resolution_y = int(param[0].split()[2])

# Initializing all the cameras' settings and reading initial condition of all the cameras
cam_num = int(param[0].split()[0])
cam_ID = []
for i in range(cam_num):
    cam_ID.append('Camera'+str(i+1))
    D.objects[cam_ID[i]].rotation_mode = 'QUATERNION'
    D.objects[cam_ID[i]].location = (float(posi_lam[i].split()[0]), float(posi_lam[i].split()[1]),\
        float(posi_lam[i].split()[2]))
    D.cameras[cam_ID[i]].lens = float(posi_lam[i].split()[3])
    D.cameras[cam_ID[i]].sensor_fit = 'HORIZONTAL'
    D.cameras[cam_ID[i]].sensor_width = float(param[0].split()[3])
    D.cameras[cam_ID[i]].sensor_height = float(param[0].split()[4])
    D.objects[cam_ID[i]].rotation_quaternion = [ float(camera[i+1].split()[3]), float(camera[i+1].split()[0]),\
            float(camera[i+1].split()[1]), float(camera[i+1].split()[2]) ]

true_posi = [-10000]
count = 0
for i in range(frame_start, frame_end):
    scene.render.resolution_percentage = 0
    count = count + 1

    bpy.context.scene.frame_current = i + 5
    bpy.ops.render.render()
    future_loc = c.dot(D.objects['Armature'].pose.bones['Hips'].head)\
       + D.objects['Armature'].location
    future_loc_tup = [future_loc[0], future_loc[1], future_loc[2]]
    D.objects["box001"].location = future_loc_tup
    while True:
        # Read file which was created by MATLAB
        try:
            # To avoid error which is occured during MATLAB is creating text file
            f12 = open(datapath + "/ell_qu_12.txt")
            k12 = open(datapath + "/ell_posi_12.txt")
            m12 = open(datapath + "ell_scale_out_12.txt")
            f23 = open(datapath + "ell_qu_23.txt")
            k23 = open(datapath + "ell_posi_23.txt")
            m23 = open(datapath + "ell_scale_out_23.txt")
            f34 = open(datapath + "ell_qu_34.txt")
            k34 = open(datapath + "ell_posi_34.txt")
            m34 = open(datapath + "ell_scale_out_34.txt")
            f45 = open(datapath + "ell_qu_45.txt")
            k45 = open(datapath + "ell_posi_45.txt")
            m45 = open(datapath + "ell_scale_out_45.txt")
            f51 = open(datapath + "ell_qu_56.txt")
            k51 = open(datapath + "ell_posi_56.txt")
            m51 = open(datapath + "ell_scale_out_56.txt")


        except IOError:
            time.sleep(0.7)
            f12 = open(datapath + "ell_qu_12.txt")
            k12 = open(datapath + "ell_posi_12.txt")
            m12 = open(datapath + "ell_scale_out_12.txt")
            f23 = open(datapath + "ell_qu_23.txt")
            k23 = open(datapath + "ell_posi_23.txt")
            m23 = open(datapath + "ell_scale_out_23.txt")
            f34 = open(datapath + "ell_qu_34.txt")
            k34 = open(datapath + "ell_posi_34.txt")
            m34 = open(datapath + "ell_scale_out_34.txt")
            f45 = open(datapath + "ell_qu_45.txt")
            k45 = open(datapath + "ell_posi_45.txt")
            m45 = open(datapath + "ell_scale_out_45.txt")
            f51 = open(datapath + "ell_qu_56.txt")
            k51 = open(datapath + "ell_posi_56.txt")
            m51 = open(datapath + "ell_scale_out_56.txt")

        ell_qu12 = f12.readlines()
        ell_posi12 = k12.readlines()
        ell_scale12 = m12.readlines()
        ell_qu23 = f23.readlines()
        ell_posi23 = k23.readlines()
        ell_scale23 = m23.readlines()
        ell_qu34 = f34.readlines()
        ell_posi34 = k34.readlines()
        ell_scale34 = m34.readlines()
        ell_qu45 = f45.readlines()
        ell_posi45 = k45.readlines()
        ell_scale45 = m45.readlines()
        ell_qu51 = f51.readlines()
        ell_posi51 = k51.readlines()
        ell_scale51 = m51.readlines()

        try:
            step = int(ell_qu23[0].split()[0])
        except IndexError:
            time.sleep(0.7)
            step = int(ell_qu23[0].split()[0])
            print (count)

        # Check whether MATLAB has updated text file
        if count == step:
            f12.close()
            k12.close()
            m12.close()
            f23.close()
            k23.close()
            m23.close()
            f34.close()
            k34.close()
            m34.close()
            f45.close()
            k45.close()
            m45.close()
            f51.close()
            k51.close()
            m51.close()
            break
        else:
            # If MATLAB has not complete to update the text file, then wait 0.5s.
            time.sleep(0.7)



    for j in range(cam_num):
#       Select object as an active object
#       bpy.context.scene.objects.active = bpy.data.objects[cam_ID[j]]
        bpy.context.scene.frame_current = i
        bpy.ops.render.render()
        D.objects["ICO1"].location = (float(ell_posi12[0].split()[0]),float(ell_posi12[1].split()[0]),float(ell_posi12[2].split()[0]))
        D.objects["ICO1"].scale = (float(ell_scale12[0].split()[0]),float(ell_scale12[1].split()[1]),float(ell_scale12[2].split()[2]))
        D.objects["ICO1"].rotation_quaternion = [ float(ell_qu12[1].split()[3]), float(ell_qu12[1].split()[0]),\
            float(ell_qu12[1].split()[1]), float(ell_qu12[1].split()[2]) ]
        D.objects["ICO2"].location = (float(ell_posi23[0].split()[0]),float(ell_posi23[1].split()[0]),float(ell_posi23[2].split()[0]))
        D.objects["ICO2"].scale = (float(ell_scale23[0].split()[0]),float(ell_scale23[1].split()[1]),float(ell_scale23[2].split()[2]))
        D.objects["ICO2"].rotation_quaternion = [ float(ell_qu23[1].split()[3]), float(ell_qu23[1].split()[0]),\
            float(ell_qu23[1].split()[1]), float(ell_qu23[1].split()[2]) ]
        D.objects["ICO3"].location = (float(ell_posi34[0].split()[0]),float(ell_posi34[1].split()[0]),float(ell_posi34[2].split()[0]))
        D.objects["ICO3"].scale = (float(ell_scale34[0].split()[0]),float(ell_scale34[1].split()[1]),float(ell_scale34[2].split()[2]))
        D.objects["ICO3"].rotation_quaternion = [ float(ell_qu34[1].split()[3]), float(ell_qu34[1].split()[0]),\
            float(ell_qu34[1].split()[1]), float(ell_qu34[1].split()[2]) ]
        D.objects["ICO4"].location = (float(ell_posi45[0].split()[0]),float(ell_posi45[1].split()[0]),float(ell_posi45[2].split()[0]))
        D.objects["ICO4"].scale = (float(ell_scale45[0].split()[0]),float(ell_scale45[1].split()[1]),float(ell_scale45[2].split()[2]))
        D.objects["ICO4"].rotation_quaternion = [ float(ell_qu45[1].split()[3]), float(ell_qu45[1].split()[0]),\
            float(ell_qu45[1].split()[1]), float(ell_qu45[1].split()[2]) ]
        D.objects["ICO5"].location = (float(ell_posi51[0].split()[0]),float(ell_posi51[1].split()[0]),float(ell_posi51[2].split()[0]))
        D.objects["ICO5"].scale = (float(ell_scale51[0].split()[0]),float(ell_scale51[1].split()[1]),float(ell_scale51[2].split()[2]))
        D.objects["ICO5"].rotation_quaternion = [ float(ell_qu51[1].split()[3]), float(ell_qu51[1].split()[0]),\
            float(ell_qu51[1].split()[1]), float(ell_qu51[1].split()[2]) ]

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0
        D.materials["ICO5"].alpha = 0

        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects[cam_ID[j]]
        bpy.ops.render.render()
        # save the image file
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format(cam_ID[j], count))
        D.materials["ICO1"].alpha = 0.5
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0
        D.materials["ICO4"].alpha = 0
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects[cam_ID[0]]
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv_Camera'+str(512), count))

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0.5
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0
        D.materials["ICO5"].alpha = 0
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects[cam_ID[1]]
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv_Camera'+str(123), count))

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0.5
        D.materials["ICO4"].alpha = 0
        D.materials["ICO4"].alpha = 0
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects[cam_ID[2]]
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv_Camera'+str(234), count))

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0.5
        D.materials["ICO5"].alpha = 0
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects[cam_ID[3]]
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv_Camera'+str(345), count))

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0
        D.materials["ICO5"].alpha = 0.5
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects[cam_ID[4]]
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv_Camera'+str(451), count))

        D.materials["ICO1"].alpha = 0.5
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0
        D.materials["ICO5"].alpha = 0
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects['Camera6']
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv512', count))

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0.5
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0
        D.materials["ICO5"].alpha = 0
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects['Camera6']
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv123', count))

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0.5
        D.materials["ICO4"].alpha = 0
        D.materials["ICO5"].alpha = 0
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects['Camera6']
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv234', count))

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0.5
        D.materials["ICO5"].alpha = 0
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects['Camera6']
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv345', count))

        D.materials["ICO1"].alpha = 0
        D.materials["ICO2"].alpha = 0
        D.materials["ICO3"].alpha = 0
        D.materials["ICO4"].alpha = 0
        D.materials["ICO5"].alpha = 0.5
        scene.render.resolution_percentage = 100
#       Select cam_ID[j] as an active camera and rendering
        bpy.context.scene.camera = bpy.data.objects['Camera6']
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath = \
            os.environ['HOMEPATH'] + output_path + '/{0}/image{1}.png'\
            .format('exv451', count))

true_posi.remove(-10000)
J = open(output_path + "/ellipsoid_posi.txt","w")
J.write("{}".format(D.objects["ICO1"].location))
J.close()
