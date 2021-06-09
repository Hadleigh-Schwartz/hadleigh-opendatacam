#connect to wifi
#make sure you have wpa_supplicant installed
#make sure u have wpa_supplicant.conf set up
echo "Connecting machine to WiFi..."
sudo pkill wpa_supplicant 
if sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf -D wext > /dev/null 2>&1; then 
	if sudo dhclient wlan0 > /dev/null 2>&1; then
		echo "Successfully connected to WiFi. "
	fi 
else
	echo "Unable to connect to Wifi"
	exit
fi 


#start container 
echo "Starting the opendatacam docker container..."
if sudo docker-compose up -d > /dev/null 2>&1; then 
	echo "Container started. At any time you may shut down the container with the command sudo docker-compose down."
else
	echo "Unable to start container"
	exit
fi



echo "Visit the server at the inet address outputted below."
ifconfig wlan0 

echo "Drag video to be annotated onto the screen. This should autatically initialize a new opendatacam session."
echo "If you don't see your video (with annotations) playing on the screen, try refreshing the page."
echo "A message will be outputted here when the video annotations are available in your mounted volume."


#make sure you have inotify-tools installed 
inotifywait -m annotations -e create -e moved_to 2> /dev/null |
    while read dir action file; do
        echo "An annotations file named '$file' just appeared in directory '$dir'"     
    done


