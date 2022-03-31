# config the proxy
hostip=$(cat /etc/resolv.conf |grep "nameserver" |cut -f 2 -d " ")
# hostip='162.105.23.22'
echo $hostip
export http_proxy="http://$hostip:7890"  # 根据实际IP和端口修改地址
export https_proxy="https://$hostip:7890"
export all_proxy="sock5://$hostip:7890"
export ALL_PROXY="sock5://$hostip:7890"
 python src/download_video.py $1 $2