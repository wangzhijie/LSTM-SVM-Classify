disp('It is running. Please wait ...');
clear all;
javaaddpath(pwd);
esmd= esmd4j.Esmd();
% load data
data=load('data\12k\007\B007-0.mat');
Y=data.X118_DE_time(1025:2048);
delt_t = 0.05; % sampling period 
t=esmd.init_t(length(Y),delt_t);
%parameters setting:
minLoop=1;
maxLoop=40;
extremeNumR=6; % >=4
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

