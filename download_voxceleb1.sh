# Download meta data for vox celeb 1
# wget -o data/ http://www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox1_dev_txt.zip

# unzip above
# unzip data/vox1_dev_txt.zip -d data/vox1_dev_txt

# make dir to place vox videos
# mkdir data/vox1_dev_vid

# loop downloaded text files 
# for FILE in $(find data/vox1_dev_txt -name '*.txt'); do\
#     python src/dl_video.py $FILE data/vox1_dev_vid

# config the proxy
hostip=$(cat /etc/resolv.conf |grep "nameserver" |cut -f 2 -d " ")
# hostip='162.105.23.22'
echo $hostip
export http_proxy="http://$hostip:7890"  # 根据实际IP和端口修改地址
export https_proxy="https://$hostip:7890"
export all_proxy="sock5://$hostip:7890"
export ALL_PROXY="sock5://$hostip:7890"
for FILE in $(find voxceleb_download/vox1_test_txt -name '*.txt'); do\
        python src/dl_video.py $FILE data/vox1_test_txt
done