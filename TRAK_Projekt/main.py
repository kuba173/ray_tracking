import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader,compileProgram
import numpy as np
import pyrr
from PIL import Image
from obj_loader import *

vertex_src="""
#version 330
layout(location=0) in vec3 v_pos;
layout(location=1) in vec2 texture;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
out vec2 v_tex;
void main()
{
gl_Position=projection*view*model*vec4(v_pos,1);
v_tex=texture;
}
"""

fragment_src="""
#version 330
out vec4 out_color;
in vec2 v_tex;
uniform sampler2D s_tex;
void main()
{
out_color=texture(s_tex,v_tex);
}
"""

def win_resize(win,width,height):
    glViewport(0,0,width,height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

if not glfw.init():
    raise Exception("glfw not initialize")

win=glfw.create_window(1280,720,"window",None,None)
if not win:
    glfw.terminate()
    raise Exception("window not created")
glfw.set_window_pos(win,300,200)
glfw.set_window_size_callback(win,win_resize)
glfw.make_context_current(win)

shaders=compileProgram(compileShader(vertex_src,GL_VERTEX_SHADER),compileShader(fragment_src,GL_FRAGMENT_SHADER))

indices,data=ObjLoader.load_model("Car.obj")

buffer=glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER,buffer)
glBufferData(GL_ARRAY_BUFFER,data.nbytes,data,GL_STATIC_DRAW)

ibuffer=glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibuffer)
glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,GL_STATIC_DRAW)

texture=glGenTextures(1)
glBindTexture(GL_TEXTURE_2D,texture)

glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT)

glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)

image=Image.open("texture.png")

image=image.transpose(Image.FLIP_TOP_BOTTOM)
image_data=image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image.width,image.height,0,GL_RGBA,GL_UNSIGNED_BYTE,image_data)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,32,ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,32,ctypes.c_void_p(12))

model_loc=glGetUniformLocation(shaders,"model")
proj_loc=glGetUniformLocation(shaders,"projection")
view_loc=glGetUniformLocation(shaders,"view")

trans_mat=pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,0]))

projection=pyrr.matrix44.create_perspective_projection_matrix(45,1280/720,0.1,100)

scaler_mat=pyrr.Matrix44.from_scale(pyrr.Vector3([3,3,3]))

view_mat=pyrr.matrix44.create_from_translation(pyrr.Vector3([0,-1,0]))

look_mat=pyrr.matrix44.create_look_at(pyrr.Vector3([-5,1.5,0]),pyrr.Vector3([0,0,0]),pyrr.Vector3([0,1,0]))
glClearColor(0,0,0.2,1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
glUseProgram(shaders)

glUniformMatrix4fv(proj_loc,1,GL_FALSE,projection)
glUniformMatrix4fv(view_loc,1,GL_FALSE,look_mat)

rotation_angles = np.array([0.0, 0.0, 0.0], dtype=np.float32)
object_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

def key_callback(window, key, scancode, action, mods):
    global object_position, rotation_angles

    if key == glfw.KEY_UP and action == glfw.PRESS:
        object_position[1] += 0.1
    elif key == glfw.KEY_DOWN and action == glfw.PRESS:
        object_position[1] -= 0.1
    elif key == glfw.KEY_LEFT and action == glfw.PRESS:
        object_position[0] -= 0.1
    elif key == glfw.KEY_RIGHT and action == glfw.PRESS:
        object_position[0] += 0.1

    elif key == glfw.KEY_X and action == glfw.PRESS:
        rotation_angles[0] += 0.1
    elif key == glfw.KEY_Y and action == glfw.PRESS:
        rotation_angles[1] += 0.1
    elif key == glfw.KEY_Z and action == glfw.PRESS:
        rotation_angles[2] += 0.1


glfw.set_key_callback(win, key_callback)
print("||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("UP Arrow Key: Moves the object up.")
print("DOWN Arrow Key: Moves the object down.")
print("LEFT Arrow Key: Moves the object to the left.")
print("RIGHT Arrow Key: Moves the object to the right.")
print("X ey: Rotates the object around the X-axis.")
print("Y Key: Rotates the object around the Y-axis.")
print("Z Key: Rotates the object around the Z-axis.")
print("||||||||||||||||||||||||||||||||||||||||||||||||||||")

while not glfw.window_should_close(win):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    rotx = pyrr.Matrix44.from_x_rotation(rotation_angles[0])
    roty = pyrr.Matrix44.from_y_rotation(rotation_angles[1])
    rotz = pyrr.Matrix44.from_z_rotation(rotation_angles[2])
    rot_mat=roty*rotx*rotz
    trans_mat = pyrr.Matrix44.from_translation(object_position)

    model_mat=pyrr.matrix44.multiply(rot_mat,trans_mat)
    glUniformMatrix4fv(model_loc,1,GL_FALSE,model_mat)
    glDrawArrays(GL_TRIANGLES,0,len(indices))

    glfw.swap_buffers(win)

glfw.terminate()