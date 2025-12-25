.SUFFIXES:

CC = nvcc
CFLAGS = -arch=native -O3
EXEC = fluid.out
PC = python3
FRAMESDIR = frames
FPS = $(shell cat fps.dat)

all: flow.png vfieldc.png transport.mp4 vfieldb.png brzeg.png

$(EXEC): fluid.cu header.cuh
	$(CC) fluid.cu -o $(EXEC) $(CFLAGS)

misc.dat psi.dat fps.dat mass.dat: $(EXEC)
	./$(EXEC)

flow.png vfieldc.png vfieldb.png brzeg.png: psi.dat navstk.py misc.dat
	$(PC) navstk.py

$(FRAMESDIR)/.frames_done: misc.dat mass.dat mass.py
	rm -f $(FRAMESDIR)/frame_*.png
	$(PC) mass.py
	@touch $(FRAMESDIR)/.frames_done

transport.mp4: $(FRAMESDIR)/.frames_done fps.dat
	ffmpeg -framerate $(FPS) -i frames/frame_%05d.png -y -c:v h264_nvenc -preset p7 -loglevel quiet -crf 18 transport.mp4

clean:
	rm -f $(EXEC)
	rm -f fps.dat mass.dat misc.dat psi.dat
	rm -f brzeg.png flow.png vfieldb.png vfieldc.png
	rm -f $(FRAMESDIR)/frame_*.png
	rm -f $(FRAMESDIR)/.frames_done
	rm -f transport.mp4

.PHONY: all clean