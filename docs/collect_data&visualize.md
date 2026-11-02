我们选择的场景：11 209 329 401，后面任务就围绕这几个场景设计。

添加了一个采集数据的脚本 `robocasa.scripts.collect_fixed_task`：

采集数据
```shell
TASK=IThorKitchenDrawerPnP
SCENE_ID=13
python -m robocasa.scripts.collect_fixed_task \
--env ${TASK} \
--scene-id ${SCENE_ID} \
--g1-control pico \
--out /path/to/datasets/${TASK}_${SCENE_ID} \
--sonic-ckpt new
```

仅可视化
```shell
TASK=IThorKitchenDrawerPnP
SCENE_ID=13
python -m robocasa.scripts.collect_fixed_task \
--env ${TASK} \
--scene-id ${SCENE_ID} \
--g1-control keyboard \
--sonic-ckpt new \
--visualize-only
```

```shell
TASK=IThorBedroomCanIntoBasket
SCENE_ID=329
python -m robocasa.scripts.visualize_scene \
--env ${TASK} \
--scene-id ${SCENE_ID} \
--scene-seed 42 \
--onscreen
```

```shell
TASK=IThorBathroomSoapToLowerShelf
SCENE_ID=401
python -m robocasa.scripts.visualize_scene \
--env ${TASK} \
--scene-id ${SCENE_ID} \
--scene-seed 42 \
--onscreen
```


这个脚本每次重置不会关掉窗口，节省了一点时间；并且固定场景 + 物品种类 + g1 初始位置 + 物品初始位置。

> ![WARNING]  
> 这个脚本还没做完，还没有适配 `IThorLivingRoomCloseLaptop` 这个任务！
