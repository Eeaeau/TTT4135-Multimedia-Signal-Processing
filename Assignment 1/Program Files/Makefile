  
compress:
	j2000_encoder -i bike.pgm -o bike_0.1976bpp.jp2 -rate 0.1976
	j2000_encoder -i bike.pgm -o bike_0.4543bpp.jp2 -rate 0.4543
	j2000_encoder -i bike.pgm -o bike_0.7071bpp.jp2 -rate 0.7071
	j2000_encoder -i bike.pgm -o bike_0.9425bpp.jp2 -rate 0.9425
	j2000_encoder -i bike.pgm -o bike_1.4361bpp.jp2 -rate 1.4361
	

decompress:
	j2000_decoder -i bike_0.1976bpp.jp2 -o bike_jp2_0.1976bpp.pgm  -Cno_speedup
	j2000_decoder -i bike_0.4543bpp.jp2 -o bike_jp2_0.4543bpp.pgm  -Cno_speedup
	j2000_decoder -i bike_0.7071bpp.jp2 -o bike_jp2_0.7071bpp.pgm  -Cno_speedup
	j2000_decoder -i bike_0.9425bpp.jp2 -o bike_jp2_0.9425bpp.pgm  -Cno_speedup
	j2000_decoder -i bike_1.4361bpp.jp2 -o bike_jp2_1.4361bpp.pgm  -Cno_speedup

.PHONY: compress decompress