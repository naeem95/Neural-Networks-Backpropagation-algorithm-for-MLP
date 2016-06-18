input=-1:0.0408:1;
targets=ones(50,1);
for i=1:50
    targets(i)=input(i)*input(i);
end
input = input';

Num_of_Hidden_Nerons = 3;
BackPropAlgo(input,targets,Num_of_Hidden_Nerons);

for i=1:50
    targets(i)=sin(input(i));
end
BackPropAlgo(input,targets,Num_of_Hidden_Nerons);

% for i=1:50
%     targets(i)=abs(input(i));
% end
% BackPropAlgo(input,targets,Num_of_Hidden_Nerons);

% for i=1:50
%     targets(i)=heaviside(input(i));
% end
% BackPropAlgo(input,targets,Num_of_Hidden_Nerons);


