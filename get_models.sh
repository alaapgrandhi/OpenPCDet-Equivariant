mkdir pretrained_models

FILE_ID_SWINT="1v74WCt4_5ubjO7PciA5T0xhQc9bz_jZu"
FILENAME_SWINT="pretrained_models/swint-nuimages-pretrained.pth"

wget --no-check-certificate \
     "https://drive.usercontent.google.com/download?id=${FILE_ID_SWINT}&confirm=t" \
     -O "${FILENAME_SWINT}"

FILE_ID_BEVFUSION="1X50b-8immqlqD8VPAUkSKI0Ls-4k37g9"
FILENAME_BEVFUSION="pretrained_models/nuscenes_bevfusion.pth"

wget --no-check-certificate \
     "https://drive.usercontent.google.com/download?id=${FILE_ID_BEVFUSION}&confirm=t" \
     -O "${FILENAME_BEVFUSION}"