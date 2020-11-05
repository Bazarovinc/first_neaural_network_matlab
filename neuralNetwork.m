classdef neuralNetwork
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        inodes;
        hnodes;
        onodes;
        lr;
        wih;
        who;
    end
    
    methods
        function obj = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
            obj.inodes = input_nodes;
            obj.hnodes = hidden_nodes;
            obj.onodes = output_nodes;
            obj.lr = learning_rate;
            obj.wih = (rand(obj.hnodes, obj.inodes) - 0.5);
            obj.who = (rand(obj.onodes, obj.hnodes) - 0.5);
        end
        
        function obj = train(obj, inputs_list, targets_list)
            inputs = inputs_list.';
            targets = targets_list.';
            hidden_inputs = obj.wih * inputs;                
            hidden_outputs = 1./(1+exp(-hidden_inputs));
            final_inputs = obj.who * hidden_outputs;
            final_outputs = 1./(1+exp(-final_inputs));
            output_errors = targets - final_outputs;
            hidden_errors = obj.who.' * output_errors;
            obj.who = obj.who + (obj.lr.*((output_errors.*final_outputs.*(1. - final_outputs)) * hidden_outputs.'));
            obj.wih = obj.wih + (obj.lr * ((hidden_errors.*hidden_outputs.* (1. - hidden_outputs)) * inputs.'));
        end
        
        function final_outputs = query(obj, inputs_list)
            inputs = inputs_list';
            hidden_inputs = obj.wih * inputs;
            hidden_outputs = 1./(1+exp(-hidden_inputs));
            final_inputs = obj.who * hidden_outputs;
            final_outputs = 1./(1+exp(-final_inputs));
        end
    end
end

