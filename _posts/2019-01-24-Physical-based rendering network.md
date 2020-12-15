# Code usage

# file structure
## Mitsuba
Mitsuba software location. It can be used to render for both python 3x and python 2.7. However, if you want to use 'import mitsuba', you have to run it in python 2.7 environment. Import thing is 'source activate setpath.sh'. The mitsuba compile installstion can be found in this [link](https://cgcvtutorials.wordpress.com/2017/05/31/install-mitsuba-on-linux-16-04/).

## planner5d
This folder stores the 3D '.obj' file and its corresponding material file '.mtl'. You can transfer the '.obj' file to '.serialize' file by running 'mtsimport main.obj main.xml'.

## util_data
* projects_list: contains all the rendering project name in a list.
* projects_list_short: contains two project to help us test our code and run demos.
* wallp_0.jpg: A sample texture file used as default texture if not assigned.
* HDR_111_Parking_Lot_2_Ref.hdr: used as environment light.
* light_geometry_compact: include the indoor light label. Three attributes are model: object id.     
  blub: the blub's material id. shade: represent the windows material id.


## rendering
### cmd_mtsb_render
Store all cmd bash file to run rendering command in bash

### project_camera
* Store all camera positions for each projects.

### project_renderingStore all rendering results for each project.

### projects_serialized
* texures: keep all texture used in this project
* main.serialized: keep the 3D structure of the scenes
* [project id].mtl: keep the material property of the scene for each 3D module.
* main_template_color.xml: the mitsuba rendering script for this project id.

# matlab
* config.m : configure all the file location: mitsuba path
* script_gernerate_serialized: Read obj file and transfer it to serialized file based on main.xml.
* script_gernerate_camera: Generate batch file for commands of all scenes
* script_mitsuba_rendering: generate mitsuba rendering script for rendering scene based on main_template_color.scene_xml
* convert_mtsb_template: generates a new xml file ready to be used by Mitsuba for rendering.
* modifyRemoveCategory: remove shape label inclued 'people' and 'plant'



## xmlModify
* modifyTwosideMaterial: replace all the BSDF type with 'twosided' in the xml file.
* modifyTransparentMaterial: modify the BSDF of all the transparent shape index with 'mask' bsdf.
* modiLightSource: put emitter to shape with bulb, and transparent material with shading.

# mitsuab_render.py
python mitsuab_render -s "scene xml file" -c "camera positions file" -o "output folder" -g "good camera index".

It will call mitsuba to run the xml script in multiprocessing.



# Questions:
1. What is the other thing in project camerafile
2. How to use '.mtl' file.
3. How to add directional light, what is the relation of ./ultils/light_geometry_compact
4. What is the semantic label of ./ultils/ModelCategoryMappingNewActive
5. what is the meaning if 'xfov_half'
6. why we need main.xml to rendering serialized file, does it include material rendering or setting?
7. what is the difference of main.xml and main_template_color.xml


# drawbacks
1. 这里边对于透明的处理就是通过mtl中的d，对应了mitsuba 中的mask特效。参数factor表示物体融入背景的数量，取值范围为0.0~1.0，取值为1.0表示完全不透明，取值为0.0时表示完全透明。当新创建一个物体时，该值默认为1.0，即无渐隐效果。与真正的透明物体材质不一样，这个渐隐效果是不依赖于物体的厚度或是否具有光谱特性。该渐隐效果对所有光照模型都有效。

2. 凡是shape_id中包含window或者包含model_xxx_mesh的，且xxx符合从ModelCategoryMappingNewActive拿出来的在window或者door这一类的透明的，都用mask把他做的透明。.mtl文件中d为0的都包含进去。但是代码中只包含了mtl对应的material id作为mask BSDF

3. Model#83_mesh 代表obj_index为83
