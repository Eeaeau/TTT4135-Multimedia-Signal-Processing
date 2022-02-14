
im_list_bike_jpg = ['bike_N5.jpg','bike_N20.jpg','bike_N40.jpg','bike_N60.jpg','bike_N80.jpg']

def get_list_of_BPP(paths):
  BPPs = []
  for path in paths:
    size_in_bytes = os.path.getsize(path)
    print(path, size_in_bytes*0.0009765625, "KB")
    size_in_bits = size_in_bytes*8
    BPPs.append(JPEG_BPP(2048, 2560, size_in_bits))

  return BPPs