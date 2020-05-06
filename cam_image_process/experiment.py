from .image_functions.image_process_pipeline import *


"""
# Channel Seletion Experiment
def channel_selection(img_raw):
    channel_R = ImgFeature3(img_raw)
    channel_S = ImgFeature3(img_raw)
    channel_R.channel_selection('R', True)
    channel_S.channel_selection('S', True)
    fc = FeatureCollector(img_raw)
    fc.add_layer("channel_R", layer=channel_R)
    fc.add_layer("channel_S", layer=channel_S)
    fc.combine("channel_R", "channel_S", "mix", (0.6, 0.9, -50))
    fc.image_show()
    fc.image_save("combined")
"""

def text_channel(img_raw):
    channel_text = ImgMask3(img_raw)
    channel_text.puttext("Hello World", show_key=True)
    fc = FeatureCollector(img_raw)
    fc.add_layer("text", layer=channel_text)
    fc.combine("main", "text", "mix", (1,1,0))
    fc.image_show()