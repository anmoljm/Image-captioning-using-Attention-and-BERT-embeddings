from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='/home/ajawalimalli/im_attention/caption_datasets/dataset_flickr8k.json',
                       image_folder='/home/ajawalimalli/im_attention/Flickr_Data/Flickr_Data/Images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/home/ajawalimalli/im_attention/outputs_imatt/',
                       max_len=50)
