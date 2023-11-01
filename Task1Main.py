import test_sam

#Main function to more easily test all 3 model types

#make sure to change file location depending on environment

test_sam.exe()
test_sam.exe(sam_checkpoint = 'Segment-Anything/checkpoints/sam_vit_b_01ec64.pth', model_type = 'vit_b')
test_sam.exe(sam_checkpoint = 'Segment-Anything/checkpoints/sam_vit_l_0b3195.pth', model_type = 'vit_l')

