clear all

writerObj = VideoWriter('20160224\smf0224return345_300frame.avi')
writerObj.FrameRate = 30;
resultname = ['smallexv2_Camera345\image',num2str(2),'.png'];
% resultname = ['big345\image',num2str(2),'.png'];
% resultname = ['image\image',num2str(2),'.png'];
% resultname = ['result1\image',num2str(2),'.png']
% resultname = ['10_Camera_before345\image',num2str(2),'.png']
imshow(resultname)
open(writerObj);
for i = 1:300
    resultname = ['smallexv2_Camera345\image',num2str(i),'.png'];
%     resultname = ['big345\image',num2str(i),'.png'];
%     resultname = ['10_Camera_before345\image',num2str(i),'.png'];
%     resultname = ['image\image',num2str(i),'.png'];
    imshow(resultname)
    frame = getframe;
    writeVideo(writerObj,frame);
end
close(writerObj);

writerObj = VideoWriter('20160224\smf0224return345_150frame.avi')
writerObj.FrameRate = 30;
resultname = ['smallexv2_Camera345\image',num2str(2),'.png'];
% resultname = ['big345\image',num2str(2),'.png'];
% resultname = ['image\image',num2str(2),'.png'];
% resultname = ['result1\image',num2str(2),'.png']
% resultname = ['10_Camera_before345\image',num2str(2),'.png']
imshow(resultname)
open(writerObj);
for i = 1:150
    resultname = ['smallexv2_Camera345\image',num2str(i),'.png'];
%     resultname = ['big345\image',num2str(i),'.png'];
%     resultname = ['10_Camera_before345\image',num2str(i),'.png'];
%     resultname = ['image\image',num2str(i),'.png'];
    imshow(resultname)
    frame = getframe;
    writeVideo(writerObj,frame);
end
close(writerObj);

writerObj = VideoWriter('20160224\smf0224return345_all_150frame.avi')
writerObj.FrameRate = 30;
resultname = ['smallexv2345\image',num2str(2),'.png'];
% resultname = ['big345\image',num2str(2),'.png'];
% resultname = ['image\image',num2str(2),'.png'];
% resultname = ['result1\image',num2str(2),'.png']
% resultname = ['10_Camera_before345\image',num2str(2),'.png']
imshow(resultname)
open(writerObj);
for i = 1:150
    resultname = ['smallexv2345\image',num2str(i),'.png'];
%     resultname = ['big345\image',num2str(i),'.png'];
%     resultname = ['10_Camera_before345\image',num2str(i),'.png'];
%     resultname = ['image\image',num2str(i),'.png'];
    imshow(resultname)
    frame = getframe;
    writeVideo(writerObj,frame);
end
close(writerObj);

writerObj = VideoWriter('20160224\smf0224return345_all_300frame.avi')
writerObj.FrameRate = 30;
resultname = ['smallexv2345\image',num2str(2),'.png'];
% resultname = ['big345\image',num2str(2),'.png'];
% resultname = ['image\image',num2str(2),'.png'];
% resultname = ['result1\image',num2str(2),'.png']
% resultname = ['10_Camera_before345\image',num2str(2),'.png']
imshow(resultname)
open(writerObj);
for i = 1:300
    resultname = ['smallexv2345\image',num2str(i),'.png'];
%     resultname = ['big345\image',num2str(i),'.png'];
%     resultname = ['10_Camera_before345\image',num2str(i),'.png'];
%     resultname = ['image\image',num2str(i),'.png'];
    imshow(resultname)
    frame = getframe;
    writeVideo(writerObj,frame);
end
close(writerObj);