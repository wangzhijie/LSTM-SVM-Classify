%parameters setting:

esmd.getSift(t, Y, optLoop, extremeNumR);
%-------------------------------------------
len = esmd.yImfsR.size();
figure(2)
for i=1:len
	subplot(len,1,i);
	plot(t,esmd.yImfsR.get(i-1));
    %y=fft(esmd.yImfsR.get(i-1),length(esmd.yImfsR.get(i-1)));
	if i==1
		ylabel('Y');
	elseif i==len
		ylabel('R');
	else
		ylabel(strcat('Imf',num2str(i-1)));
	end
end
m1=esmd.yImfsR.get(0);
m2=esmd.yImfsR.get(1);
m3=esmd.yImfsR.get(2);
m4=esmd.yImfsR.get(3);
for i=1:1024
  sum=0;
  for j=0:3
      m=esmd.yImfsR.get(j);
      sum=sum+m(i)*m(i);
  end
  sum=sqrt(sum);
  m1(i)=abs(m1(i))/sum;
  m2(i)=abs(m2(i))/sum;
  m3(i)=abs(m3(i))/sum;
  m4(i)=abs(m4(i))/sum;
end
%存储归一化特征向量值到excel中
data=[m1,m2,m3,m4];
s=xlswrite('tezhengxiangliang\B007.xlsx',data,2);


%-------------------------------------------
% len=esmd.interfs.size();
% figure(3)
% frequency distribution figure
% for i=1:len
% 	plot(t,esmd.interfs.get(i-1))
% 	hold on
% end
% 
% -------------------------------------------
% figure(4)
% frequency Amplitude figure
% for i=1:len
% 	subplot(2*len,1,2*i-1)
% 	plot(t,esmd.interfs.get(i-1))
% 	ylabel(strcat('F',num2str(i)))
% 
% 	subplot(2*len,1,2*i)
% 	plot(t,esmd.upperEvelops.get(i-1))
% 	ylabel(strcat('A',num2str(i)))
% end
% 
% -------------------------------------------
% figure(5)
% AdaptGlobalMeanCurve on Y
% len = esmd.yImfsR.size();
% plot(t,Y);
% hold on;
% R=esmd.yImfsR.get(len-1);
% plot(t,R);
% 
% -------------------------------------------
% figure(6)
% plot Y-R  
% plot(t,Y-R)
% 
% -------------------------------------------
% figure(7)
% plot(t,esmd.energy)
% 
% clear all;
% javarmpath(pwd);
