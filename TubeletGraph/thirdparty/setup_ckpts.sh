if [ ! -d "./_ckpts" ]; then
    mkdir ./_ckpts
fi

if [ ! -f "./_ckpts/CropFormer_hornet_3x_03823a.pth" ]; then
    echo "Downloading CropFormer checkpoint ..."
    python -m gdown 1b36D3PEh2TqzqXBCDhTySHmla2MltXVU -O ./_ckpts/CropFormer_hornet_3x_03823a.pth
fi

if [ ! -f "./_ckpts/sam2.1_hiera_large.pt" ]; then
    echo "Downloading SAM2.1 checkpoint ..."
    python -m gdown 1f4rcOsJGj_Mi3laow8gkb-X8G2Ud1z_C -O ./_ckpts/sam2.1_hiera_large.pt
fi

if [ ! -f "./_ckpts/fcclip_cocopan.pth" ]; then
    echo "Downloading FC-CLIP checkpoint ..."
    python -m gdown 1iRtAUrOWKVHeXFgr-EFxozhQLhiIZVjT -O ./_ckpts/fcclip_cocopan.pth
fi