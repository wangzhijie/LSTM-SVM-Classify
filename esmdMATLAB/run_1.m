disp('It is running. Please wait ...');
clear all;
x1=50;
javaaddpath(pwd);
esmd= esmd4j.Esmd();
% load data
data=load('data\12k\021\OR021@6-2.mat');
Y=data.X236_DE_time((x1-1)*1024+1:(x1-1)*1024+1024);
delt_t = 0.05; % sampling period 
t=esmd.init_t(length(Y),delt_t);
%parameters setting:
minLoop=1;
maxLoop=40;
extremeNumR=7; % >=4
jianGeNum = 1; %
rList = esmd.getVarianceRatio(t, Y, minLoop, maxLoop, extremeNumR);
[minVar,idx]=min(rList);
optLoop=idx+minLoop-1;
%disp('optimal loop is: ');
%fprintf('optLoop=%d\n',optLoop);
%-----------------------------------
x=minLoop:maxLoop;
%figure(1)
%plot(x,rList)
run('run_2.m');
