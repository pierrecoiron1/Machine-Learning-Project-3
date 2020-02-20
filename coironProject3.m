%%Project by: Pierre Coiron
clear all;
clc;
%%
%%Meta Variables
numOfHiddenLayers=700;
eta=0.01;
numOfTrainingSteps=500000;

% numOfHiddenLayers=500;
% eta=0.01;
% numOfTrainingSteps=100000;

%%
%%Generate test Sampels
sampleInputData(1,:)=-8:0.2:8;
sampleInputData(2,:)=-8:0.2:8;

[xTrainMesh, YTrainMesh]=meshgrid(sampleInputData(1,:),sampleInputData(2,:));

%%get the size of both test samples
sampSize=size(sampleInputData);
sampSize=sampSize(2);
sampleOutputData=ones(sampSize,sampSize);

%output data
sampleOutputData=sin(sqrt(xTrainMesh.^2+YTrainMesh.^2))./sqrt(xTrainMesh.^2+YTrainMesh.^2);
sampleOutputData(41,41)=1;
%components of output data, to be used later
maxSampleOutput=max(max(sampleOutputData));
minSampleOutput=min(min(sampleOutputData));

%%normalize test samples
[sampleInputNorm,ps] = mapminmax(sampleInputData, 0, 1); % normalize the training samples
[SampleOutputNorm,ts] = mapminmax(sampleOutputData, 0, 1);


%%
%%Build Stuff
%%Constuct the Network
inputTest=ones(2,1);                         %   the single input neuron
targetOutput=ones(2,1);                       %   the target for each input
hiddenOutputTest=ones(numOfHiddenLayers,1);  %   the outputs of the hidden neurons
outputTest=ones(1,1);                        %   the single output neuron 

%initialize weights
hiddenInputWeight=ones(2,numOfHiddenLayers);
hiddenOutputWeight=ones(numOfHiddenLayers,1);

%initialize error
hiddenInputError=ones(1,numOfHiddenLayers);
hiddenOutputError=ones(numOfHiddenLayers,1);
%%
%start w using a randum number within (-0.01,0.01)
for inputNeuronPosition=1:2       %initialize w using a random number within (-0.01, 0.01) 
   for hiddenLayersPosition=1:numOfHiddenLayers
       hiddenInputWeight(inputNeuronPosition,hiddenLayersPosition)=(0.01-(-0.01))*rand()+(-0.01);
   end
end

%start v using a randum number within (-0.01,0.01)
for hiddenLayersPosition=1:numOfHiddenLayers       
   for outputNeuronPosition=1:1
       hiddenOutputWeight(hiddenLayersPosition,outputNeuronPosition)=(0.01-(-0.01))*rand()+(-0.01);
   end
end

%% %BP algorithm
% training the neural network for 20,000 times
count=0;
numberofOutputNeurons=1;
for j=1:numOfTrainingSteps

%take all training samples into the BP algorithm
    err=0;   
    for t=1:sampSize
        %%generate random sample value
        xSampledIndex=(sampSize-1)*rand()+1; % generate a random number between 1 and the sampleSize
        xSampledIndex=round(xSampledIndex);
        
        ySampledIndex=(sampSize-1)*rand()+1; % generate a random number between 1 and the sampleSize
        ySampledIndex=round(ySampledIndex);
        
        %%pull random values based on random index
        inputTest(1,1)=sampleInputNorm(1,xSampledIndex);
        inputTest(2,1)=sampleInputNorm(2,ySampledIndex);
        targetOutput=SampleOutputNorm(xSampledIndex,ySampledIndex);
        
        %calculate the outputs of the hidden layer
        for hiddenLayerPosition=1:numOfHiddenLayers
            xSampleWeight=hiddenInputWeight(1,hiddenLayerPosition);
            ySampleWeight=hiddenInputWeight(2,hiddenLayerPosition);
            % the sigmoid fuction here
            hiddenOutputTest(hiddenLayerPosition,1)=1/(1+exp(-(xSampleWeight'*inputTest(1,1)+ySampleWeight'*inputTest(2,1)))); 
        end
        
        %calculate the output of the output layer: y
        for outputIndex=1:1
            outputTest(outputIndex)=hiddenOutputWeight(:,outputIndex)'*hiddenOutputTest;
            err=err+abs(targetOutput-outputTest(outputIndex));  % calculate the error
        end
        
        %calculate delta_v
        for OutputNeuronIndex=1:1
             hiddenOutputError(:,OutputNeuronIndex)=eta*(targetOutput-outputTest(numberofOutputNeurons))*hiddenOutputTest;
        end
   
        %calculate delta_w
        for hiddenLayerPosition=1:numOfHiddenLayers
            sum=0;
            for numberofOutputNeurons=1:1
                sum=sum+(targetOutput(numberofOutputNeurons)-outputTest(numberofOutputNeurons))*hiddenOutputWeight(hiddenLayerPosition,numberofOutputNeurons);
            end
            hiddenInputError(1,hiddenLayerPosition)=eta*sum*hiddenOutputTest(hiddenLayerPosition)*(1-hiddenOutputTest(hiddenLayerPosition))*inputTest(1,:);
            hiddenInputError(2,hiddenLayerPosition)=eta*sum*hiddenOutputTest(hiddenLayerPosition)*(1-hiddenOutputTest(hiddenLayerPosition))*inputTest(2,:);
        end
 
        % update v
        hiddenOutputWeight=hiddenOutputWeight+hiddenOutputError;
        % update w
        hiddenInputWeight=hiddenInputWeight+hiddenInputError;
        
        % save the history of v(10,1) and w(1,15) 
        count=count+1;
        v_history(count)=hiddenOutputWeight(1,1);
        w_history(count)=hiddenInputWeight(1,1);
    end
    err_history(j)=err/(sampSize*1.0); % save the history of error.
    
    disp(err + " " + j)
end
%%
%%training results
plot(v_history);
legend ('Weight from Hidden Cell 1 to Output');
figure;
plot(w_history);
legend('Weight from x input to hidden cell 1)');
figure;
plot(err_history);
legend('Error History');
%%
%%% test the network

%generate final test data
xfinalRealTestInputData=-8:0.2:8;
yfinalRealTestInputData=-8:0.2:8;
%%number of data points
xFinalRealTestInputDataSize=size(yfinalRealTestInputData);
xFinalRealTestInputDataSize=xFinalRealTestInputDataSize(2);
yFinalRealTestInputDataSize=xFinalRealTestInputDataSize;

%Meshfrid for final 
[xfinalRealTestInpuDatatMesh,yfinalRealTestInputMeshData]=meshgrid(xfinalRealTestInputData,yfinalRealTestInputData);

%calculate real final output
finalRealTestOutput=sin(sqrt(xfinalRealTestInpuDatatMesh.^2+yfinalRealTestInputMeshData.^2))./sqrt(xfinalRealTestInpuDatatMesh.^2+yfinalRealTestInputMeshData.^2);
%real final components
maxFinalRealTestOutput=max(max(finalRealTestOutput));
minFinalRealTestOutput=min(min(finalRealTestOutput));

%normalize the the real inputs
NormalizedTestInput(:,1) = mapminmax(xfinalRealTestInputData,0,1);
NormalizedTestInput(:,2) = mapminmax(yfinalRealTestInputData,0,1);
%normalize the real outputs
finalNormalTestOutput=mapminmax(finalRealTestOutput,0,1);

%finding the value of the theortical output
for xFinalRealTestInputDataIndex=1:xFinalRealTestInputDataSize
    for yFinalRealTestInputDataIndex=1:yFinalRealTestInputDataSize
        %establishing input vector
        testedInputVector(1,1)=xfinalRealTestInputData(xFinalRealTestInputDataIndex);
        testedInputVector(2,1)=xfinalRealTestInputData(yFinalRealTestInputDataIndex);
        %calculating hidden output
        testedHiddenInputVector=1./(1+exp(-(hiddenInputWeight'*testedInputVector)));
        %calculating final output
        finalNormalizedOutput(xFinalRealTestInputDataIndex,yFinalRealTestInputDataIndex)=hiddenOutputWeight(:,1)'*testedHiddenInputVector;
    end
end

%normalize output
finalRealOutput=mapminmax(finalNormalizedOutput,minFinalRealTestOutput,maxFinalRealTestOutput);

%plot this
figure
surf(xfinalRealTestInputData,yfinalRealTestInputData,finalRealOutput,'edgecolor','g');
hold on
surf(xfinalRealTestInputData,yfinalRealTestInputData,finalRealTestOutput,'edgecolor','r');



% plot(xNormalizedShow,yNormalizedShow,FinalOutput, zProof,'b');
% legend('Predicted', 'Actual');
% xlabel('x')
% ylabel('y')
% title('y=x^2')