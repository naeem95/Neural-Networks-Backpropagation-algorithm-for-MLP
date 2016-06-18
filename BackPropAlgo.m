function BackPropAlgo(input, targets,Num_of_Hidden_Nerons)

w_i=rand(Num_of_Hidden_Nerons,2);
w_j=rand(Num_of_Hidden_Nerons+1,1);

N=size(input,1);
y=ones(N,1);
step_size=0.001;

b_input=[input ones(N,1)];
t_b_input=b_input';

errorValue = 1;
Divergence_Criteria = 0.05;

while errorValue > Divergence_Criteria
    error_w_ji=zeros(Num_of_Hidden_Nerons,2);
    error_w_kj=zeros(Num_of_Hidden_Nerons+1,1);   
        for i=1:N          
            a_j = w_i*t_b_input(:,i);
            Output_of_HiddenLayer=Tanh(a_j);
            b_Output_of_HiddenLayer=[Output_of_HiddenLayer;1];
            y(i)=sum(w_j.*b_Output_of_HiddenLayer);
            
            delta_k=y(i)-targets(i);           
            gradient_kj=delta_k*b_Output_of_HiddenLayer;
            error_w_kj=error_w_kj+gradient_kj;
            
            delta_j=(1-(b_Output_of_HiddenLayer.*b_Output_of_HiddenLayer)).*(delta_k*w_j);
            gradient_ji=delta_j(1:Num_of_Hidden_Nerons)*t_b_input(:,i)';
            error_w_ji=error_w_ji+gradient_ji;           
        end
        
        difference = targets - y; 
        square = difference.*difference;
        errorValue = sqrt(sum(square(:)));
        errorValue
        w_i=w_i - step_size*error_w_ji;
        w_j=w_j - step_size*error_w_kj;    
end
    clear y;
    y=ones(N,1);
    
    hidden_Output_1 = ones(N,1);
    hidden_Output_2 = ones(N,1);
    hidden_Output_3 = ones(N,1);

    for i=1:N
        a_j = w_i*t_b_input(:,i);
        Output_of_HiddenLayer=Tanh(a_j);
        hidden_Output_1(i)=Output_of_HiddenLayer(1);
        hidden_Output_2(i)=Output_of_HiddenLayer(2);
        hidden_Output_3(i)=Output_of_HiddenLayer(3);
        b_Output_of_HiddenLayer=[Output_of_HiddenLayer;1];
        y(i)=sum(sum(w_j.*b_Output_of_HiddenLayer));
    end
    
    hold on;
    plot(t_b_input(1,:),y,'b.');
    plot(t_b_input(1,:),targets,'r','Linewidth',1);
    plot(t_b_input(1,:),hidden_Output_1,'--');
    plot(t_b_input(1,:),hidden_Output_2,'--');
    plot(t_b_input(1,:),hidden_Output_3,'--');
    hold off;
end
