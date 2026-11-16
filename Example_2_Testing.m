%% Paper Title: Nonlinear System Identification Using Function-Link Recurrent General Type-2 Fuzzy Neural Networks: A Comparative Study
% Authors: Ahmad M. El-Nagar, Raouf Fareh, Sofiane Khadraoui, Maamar
% Bettayeb, Tamer Rabie, Ibrahim Kamel, A. Aziz Khater
%=========================================================================
%% Example 2: Time-varying system
% Testing Program and Plotting Fig. 5
%=========================================================================
NT=1200;
%=======================
n=3;             % number of inputs
% parameters of GT2FS
AA(1)=0.0;
ww(1)=1;
%=================================
               % Model Parameters
for i=1:M
    for j=1:n
        m1(1,i,j)=m1(N,i,j);
        m2(1,i,j)=m2(N,i,j);
        d(1,i,j)=d(N,i,j);
    end
end
GL(1)=0.9;
GR(1)=0.1;
for i=1:M
FUo(i)=0.0;
FLo(i)=0.0;
end
etaV(1)=etaV(N);
for i=1:M
     for p=1:Ip
         WEI_L(1,i,p)=WEI_L(N,i,p);   
         WEI_R(1,i,p)=WEI_R(N,i,p); 
     end
  end
%==================================
yp(1)=0; yp(2)=0; yp(3)=0;
tic
for k=1:NT
    if k < 250
        u(k+2)=sin(pi*(k+2)/25);
    else if k >=250 && k < 500
            u(k+2)=1;
        else if k >=500 && k <750
                u(k+2)=-1;
            else
                u(k+2)=0.3*sin(pi*(k+2)/25)+0.1*sin(pi*(k+2)/32)+0.6*sin(pi*(k+2)/10);
            end
        end
    end
    % System
    aa(k)=1.2-0.2*cos(2*pi*k/NT);
    bb(k)=1-0.4*sin(2*pi*k/NT);
    cc(k)=1+0.4*sin(2*pi*k/NT);
    yp(k+3)=(yp(k+2)*yp(k+1)*yp(k)*u(k+1)*(yp(k)-bb(k))+cc(k)*u(k+2))/(aa(k)+yp(k)*yp(k)+yp(k+1)*yp(k+1));
    %========================================================================================
    % Model
    % Layer 1:
  x(1)=u(k+2);
  x(2)=yp(k+2);
  x(3)=yp(k+1);
  %===================
% Layer 2:
  for i=1:M
      for j=1:n
          if x(j) < m1(k,i,j)
              MU(k,i,j)=exp(-(((x(j)-m1(k,i,j))^2)/(d(k,i,j)^2)));
          else if x(j) >= m1(k,i,j) && x(j) <= m2(k,i,j)
                  MU(k,i,j)=1;
              else if x(j) > m2(k,i,j)
                      MU(k,i,j)=exp(-(((x(j)-m2(k,i,j))^2)/(d(k,i,j)^2)));
                  else
                      MU(k,i,j)=0;
                  end
              end
          end
          if x(j) <= (m1(k,i,j)+m2(k,i,j))/2
              ML(k,i,j)=exp(-(((x(j)-m2(k,i,j))^2)/(d(k,i,j)^2)));
          else if x(j) > (m1(k,i,j)+m2(k,i,j))/2
                  ML(k,i,j)=exp(-(((x(j)-m1(k,i,j))^2)/(d(k,i,j)^2)));
              else
                  ML(k,i,j)=0;
              end
          end
         % caclute apex of alpha - triangle
        %  MLA(k,i,j)=ML(k,i,j)+ww(k)*(MU(k,i,j)-ML(k,i,j))*AA(k);
        % MUA(k,i,j)=MU(k,i,j)-(1-ww(k))*(MU(k,i,j)-ML(k,i,j))*AA(k);
        % 
         % caclute apex of alpha - trapezoid
         MLA(k,i,j)=ML(k,i,j)+0.5*ww(k)*(MU(k,i,j)-ML(k,i,j))*AA(k);
         MUA(k,i,j)=MU(k,i,j)-0.5*(MU(k,i,j)-ML(k,i,j))*AA(k);  
      end
  end
  %=================================================================
  % Layer 3:
  for i=1:M
      FU(i)=etaV(k)*(MUA(k,i,1)*MUA(k,i,2)*MUA(k,i,3))+(1-etaV(k))*FUo(i);
      FL(i)=etaV(k)*(MLA(k,i,1)*MLA(k,i,2)*MLA(k,i,3))+(1-etaV(k))*FLo(i);
      FC(i)=(FU(i)+FL(i))/2;
  end
  %====================================================================
  % Layer 4: Function Link Network
 % Inputs for FLN
 Ix(1)=x(1);
 Ix(2)=sin(pi*x(1));
 Ix(3)=cos(pi*x(1));
 Ix(4)=x(2);
 Ix(5)=sin(pi*x(2));
 Ix(6)=cos(pi*x(2));
 Ix(7)=x(3);
 Ix(8)=sin(pi*x(3));
 Ix(9)=cos(pi*x(3));
 Ix(10)=x(1)*x(2);
 Ix(11)=x(1)*x(3);
 Ix(12)=x(2)*x(3);
 Ix(13)=x(1)*x(2)*x(3);
 % Number of nodes in each layer
 Ip=13;           % number of inputs
 RH=2;           % number of outputs
 % initial values for weights
  % for rh=1:RH
     for p=1:Ip
         WEI_L(1,1,p)=yp(3)-0.01;   
         WEI_R(1,1,p)=yp(3)+0.01; 
     end
  % end
  % output layer
for i=1:M
     for p=1:Ip
         EX_L(i,p)=WEI_L(k,i,p)*Ix(p);
         EX_R(i,p)=WEI_R(k,i,p)*Ix(p);
     end
  WL(k,i)=sum(EX_L(i,:));
  WR(k,i)=sum(EX_R(i,:));
 end
  %====================================================================
  % Layer 5:
  for i=1:M
      ss1(i)=FU(i)*WL(k,i);
      ss2(i)=FL(i)*WL(k,i);
      ss3(i)=FL(i)+FU(i);
      ss4(i)=FL(i)*WR(k,i);
      ss5(i)=FU(i)*WR(k,i);
  end
  ss11=sum(ss1);
  ss22=sum(ss2);
  ss33=sum(ss3);
  ss44=sum(ss4);
  ss55=sum(ss5);
  YL=((1-GL(k))*ss11+GL(k)*ss22)/ss33;
  YR=((1-GR(k))*ss44+GR(k)*ss55)/ss33;
  ynn=(YL+YR);
  %===================================================================
  % Layer 6:
  yn(k+3)=ynn;
  %========================================================================
  %========================================================================
                     % Update Parameters (GM)
el(k+3)=yn(k+3)-yp(k+3);
RMSE_k(3)=0;
RMSE_k(k+3)=sqrt((el(k+3)^2)/k);
er(k+3)=(RMSE_k(k+3)-RMSE_k(k+2))/RMSE_k(k+3);
LL(1)=0.05;           
LLW(1)=0.35;         
LLeta(1)=0.6;      
LLA(1)=0.1;                                  
LLWW(1)=0.1;      

meo1=0.01;      
meo2=0.0001;      
meo3=0.01;      
meo4=0.01;     
meo5=0.001;     
if (el(k+3))>=0.001    
if er(k+3)<0
LL(k+1)=LL(k)*(1+meo1*exp(-(er(k+3)^2)/2));
LLW(k+1)=LLW(k)*(1+meo2*exp(-(er(k+3)^2)/2));
LLeta(k+1)=LLeta(k)*(1+meo3*exp(-(er(k+3)^2)/2));
LLA(k+1)=LLA(k)*(1+meo4*exp(-(er(k+3)^2)/2));
LLWW(k+1)=LLWW(k)*(1+meo5*exp(-(er(k+3)^2)/2));
else
LL(k+1)=LL(k)*(1-meo1*exp(-(er(k+3)^2)/2));
LLW(k+1)=LLW(k)*(1-meo2*exp(-(er(k+3)^2)/2));
LLeta(k+1)=LLeta(k)*(1-meo3*exp(-(er(k+3)^2)/2));
LLA(k+1)=LLA(k)*(1-meo4*exp(-(er(k+3)^2)/2));
LLWW(k+1)=LLWW(k)*(1-meo5*exp(-(er(k+3)^2)/2));
end
else
LL(k+1)=LL(k);
LLW(k+1)=LLW(k);
LLeta(k+1)=LLeta(k);
LLA(k+1)=LLA(k);
LLWW(k+1)=LLWW(k);
end
  %====================================================
  for i=1:M
      for j=1:n
          if x(j)<=m1(k,i,j)
              PFUm1(k,i,j)=FU(i)*((x(j)-m1(k,i,j))/((d(k,i,j))^2));
          else
              PFUm1(k,i,j)=0;
          end
          if x(j)> (m1(k,i,j)+m2(k,i,j))/2
              PFLm1(k,i,j)=FL(i)*((x(j)-m1(k,i,j))/((d(k,i,j))^2));
          else
              PFLm1(k,i,j)=0;
          end
          if x(j)> m2(k,i,j)
              PFUm2(k,i,j)=FU(i)*((x(j)-m2(k,i,j))/((d(k,i,j))^2));
          else
              PFUm2(k,i,j)=0;
          end
          if x(j) <= (m1(k,i,j)+m2(k,i,j))/2
              PFLm2(k,i,j)=FL(i)*((x(j)-m2(k,i,j))/((d(k,i,j))^2));
          else
              PFLm2(k,i,j)=0;
          end
          if x(j) < m1(k,i,j)
              PFUd(k,i,j)=FU(i)*(((x(j)-m1(k,i,j))^2)/((d(k,i,j))^3));
          else if x(j) > m2(k,i,j)
                  PFUd(k,i,j)=FU(i)*(((x(j)-m2(k,i,j))^2)/((d(k,i,j))^3));
              else
                  PFUd(k,i,j)=0;
              end
          end
          if x(j) <= (m1(k,i,j)+m2(k,i,j))/2
              PFLd(k,i,j)=FL(i)*(((x(j)-m2(k,i,j))^2)/((d(k,i,j))^3));
          else if x(j) > (m1(k,i,j)+m2(k,i,j))/2
                  PFLd(k,i,j)=FL(i)*(((x(j)-m1(k,i,j))^2)/((d(k,i,j))^3));
              end
          end
 % apex of alpha - trangle
           % PMAUMU(k,i,j)=1-(1-ww(k))*AA(k);
           % PMAUML(k,i,j)=(1-ww(k))*AA(k);
           % PMALMU(k,i,j)=ww(k)*AA(k);
           % PMALML(k,i,j)=1-ww(k)*AA(k);
           % 
           %   PMUA_A(k,i,j)=(1-ww(k))*(MU(k,i,j)-ML(k,i,j));
           %   PMLA_A(k,i,j)=ww(k)*(MU(k,i,j)-ML(k,i,j));
           %   PMUA_w(k,i,j)=(MU(k,i,j)-ML(k,i,j))*AA(k);
           %   PMLA_w(k,i,j)=(MU(k,i,j)-ML(k,i,j))*AA(k);
  
 % apex of alpha - trapezoid
           PMAUMU(k,i,j)=1-0.5*ww(k)*AA(k);
           PMAUML(k,i,j)=0.5*ww(k)*AA(k);
           PMALMU(k,i,j)=0.5*ww(k)*AA(k);
           PMALML(k,i,j)=1-0.5*ww(k)*AA(k);

           PMUA_A(k,i,j)=-0.5*(MU(k,i,j)-ML(k,i,j));
           PMLA_A(k,i,j)=0.5*ww(k)*(MU(k,i,j)-ML(k,i,j));
           PMUA_w(k,i,j)=0;
           PMLA_w(k,i,j)=0.5*(MU(k,i,j)-ML(k,i,j))*AA(k);

      end
      PFUMAU(k,i,1)=etaV(k)*(MUA(k,i,2)*MUA(k,i,3));
      PFUMAU(k,i,2)=etaV(k)*(MUA(k,i,1)*MUA(k,i,3));
      PFUMAU(k,i,3)=etaV(k)*(MUA(k,i,1)*MUA(k,i,2));
      PFLMAL(k,i,1)=etaV(k)*(MLA(k,i,2)*MLA(k,i,3));
      PFLMAL(k,i,2)=etaV(k)*(MLA(k,i,1)*MLA(k,i,3));
      PFLMAL(k,i,3)=etaV(k)*(MLA(k,i,1)*MLA(k,i,2));
for j=1:n
       Dm1(k,i,j)=(((((1-GL(k))*WL(i)-YL)/(ss33))+((GR(k)*WR(i)-YR)/(ss33)))*(PFUMAU(k,i,j)*((PMAUMU(k,i,j)*PFUm1(k,i,j))+(PMAUML(k,i,j)*PFLm1(k,i,j))))+(((GL(k)*WL(i)-YL)/(ss33))+(((1-GR(k))*WR(i)-YR)/(ss33)))*(PFLMAL(k,i,j)*((PMALMU(k,i,j)*PFUm1(k,i,j))+(PMALML(k,i,j)*PFLm1(k,i,j)))));
       Dm2(k,i,j)=(((((1-GL(k))*WL(i)-YL)/(ss33))+((GR(k)*WR(i)-YR)/(ss33)))*(PFUMAU(k,i,j)*((PMAUMU(k,i,j)*PFUm2(k,i,j))+(PMAUML(k,i,j)*PFLm2(k,i,j))))+(((GL(k)*WL(i)-YL)/(ss33))+(((1-GR(k))*WR(i)-YR)/(ss33)))*(PFLMAL(k,i,j)*((PMALMU(k,i,j)*PFUm2(k,i,j))+(PMALML(k,i,j)*PFLm2(k,i,j)))));
       Dd(k,i,j)=(((((1-GL(k))*WL(i)-YL)/(ss33))+((GR(k)*WR(i)-YR)/(ss33)))*(PFUMAU(k,i,j)*((PMAUMU(k,i,j)*PFUd(k,i,j))+(PMAUML(k,i,j)*PFLd(k,i,j))))+(((GL(k)*WL(i)-YL)/(ss33))+(((1-GR(k))*WR(i)-YR)/(ss33)))*(PFLMAL(k,i,j)*((PMALMU(k,i,j)*PFUd(k,i,j))+(PMALML(k,i,j)*PFLd(k,i,j)))));
       DA(k,i,j)=(((((1-GL(k))*WL(i)-YL)/(ss33))+((GR(k)*WR(i)-YR)/(ss33)))*(PFUMAU(k,i,j)*((PMUA_A(k,i,j))))+(((GL(k)*WL(i)-YL)/(ss33))+(((1-GR(k))*WR(i)-YR)/(ss33)))*(PFLMAL(k,i,j)*((PMLA_A(k,i,j)))));
       Dw(k,i,j)=(((((1-GL(k))*WL(i)-YL)/(ss33))+((GR(k)*WR(i)-YR)/(ss33)))*(PFUMAU(k,i,j)*((PMUA_w(k,i,j))))+(((GL(k)*WL(i)-YL)/(ss33))+(((1-GR(k))*WR(i)-YR)/(ss33)))*(PFLMAL(k,i,j)*((PMLA_w(k,i,j)))));
end
SDA(k,i)=sum(DA(k,i,:));
SDw(k,i)=sum(Dw(k,i,:));
for p=1:Ip
DWEI_L(k,i,p)=(1/ss33)*(((1-GL(k))*FU(i)+GL(k)*FL(i))*Ix(p));
DWEI_R(k,i,p)=(1/ss33)*(((1-GR(k))*FL(i)+GR(k)*FU(i))*Ix(p));
end
Detavi(i)=((((((1-GL(k))*WL(i)-YL)/(ss33))+((GR(k)*WR(i)-YR)/(ss33)))*(MUA(k,i,1)*MUA(k,i,2)*MUA(k,i,3))+(((GL(k)*WL(i)-YL)/(ss33))+(((1-GR(k))*WR(i)-YR)/(ss33)))*(MLA(k,i,1)*MLA(k,i,2))));
% end   % end for a=2:num
 SDetavi(i)=Detavi(i);
       for j=1:n
            if  el(k+3) > 0.00001
               m1(k+1,i,j)=m1(k,i,j)-LL(k+1)*((m1(k,i,j)+(el(k+3))*Dm1(k,i,j))/(1+(Dm1(k,i,j))^2));
           else
               m1(k+1,i,j)=m1(k,i,j);
            end
                if  el(k+3) > 0.0001
               m2(k+1,i,j)=m2(k,i,j)-LL(k+1)*((m2(k,i,j)+(el(k+3))*Dm2(k,i,j))/(1+(Dm2(k,i,j))^2));
           else
               m2(k+1,i,j)=m2(k,i,j);
           end
          d(k+1,i,j)=d(k,i,j)-LL(k+1)*(el(k+3))*Dd(k,i,j);
       end
     for p=1:Ip
         WEI_L(k+1,i,p)=WEI_L(k,i,p)-LLW(k+1)*(el(k+3))*DWEI_L(k,i,p);
         WEI_R(k+1,i,p)=WEI_R(k,i,p)-LLW(k+1)*(el(k+3))*DWEI_R(k,i,p);
     end
  % update eta for recurrent
  end   % end of for i=1:M
     if sum(SDetavi) > 1.0
               etaV(k+1)=etaV(k)-LL(k+1)*((etaV(k)+(el(k+3))*sum(SDetavi))/(1+(sum(SDetavi))^2));
           else
               etaV(k+1)=etaV(k);
     end
    if etaV(k+1) > 1
      etaV(k+1)=1;
  else if etaV(k+1) < 0
          etaV(k+1)=0;
      else
          etaV(k+1)=etaV(k+1);
      end
  end
    SSDA(k)=sum(SDA(k,:));
    SSDw(k)=sum(SDw(k,:));
   AA(k+1)=AA(k)-LLA(k+1)*(el(k+3))*SSDA(k);
   ww(k+1)=ww(k)-LLWW(k+1)*(el(k+3))*SSDw(k);
   if AA(k+1) > 1
      AA(k+1)=1;
  else if AA(k+1) < 0
          AA(k+1)=0;
      else
          AA(k+1)=AA(k+1);
      end
   end
   if ww(k+1) > 1
      ww(k+1)=1;
  else if ww(k+1) < 0
          ww(k+1)=0;
      else
          ww(k+1)=ww(k+1);
      end
   end
  for i=1:M
      NN1(i)=WL(i)*(FL(i)-FU(i));
      NN2(i)=WR(i)*(FU(i)-FL(i));
  end
  SNN1=sum(NN1);
  SNN2=sum(NN2);
  DGL(k)=(SNN1/ss33);
  DGR(k)=(SNN2/ss33);
      GL(k+1)=GL(k)-LL(k+1)*(el(k+3))*DGL(k);
         if el(k+3) > 0.00001
               GR(k+1)=GR(k)-LL(k+1)*((GR(k)+(el(k+3))*DGR(k))/(1+(DGR(k))^2));
           else
               GR(k+1)=GR(k);
           end
  if GL(k+1) > 1
      GL(k+1)=1;
  else if GL(k+1) < 0
          GL(k+1)=0;
      else
          GL(k+1)=GL(k+1);
      end
  end
  if GR(k+1) > 1
      GR(k+1)=1;
  else if GR(k+1) < 0
          GR(k+1)=0;
      else
          GR(k+1)=GR(k+1);
      end
  end
  % update variable for recurrent
    for i=1:M
       FUo(i)=FU(i);
       FLo(i)=FL(i);
    end
  %========================================================================
  %========================================================================
                        
 e(k+3)=yn(k+3)-yp(k+3);
   ee(k+3)=e(k+3)*e(k+3);
end
toc
tt=toc
RMSE=sqrt(sum(ee)/NT)
  k=1:NT;
  plot(k,yp(k+3),"k",k,yn(k+3),"b")  
    xlabel("Samples")
  ylabel("System output")
  legend("Desired output","FL-RGT2FNN")
  
  
          