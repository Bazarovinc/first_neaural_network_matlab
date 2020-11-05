clear all;
input_nodes = 784;
hidden_nodes = 500;
output_nodes = 10;
learning_rate = 0.2;
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);
db = readmatrix("mnist_train.csv");
for i= 1:length(db)
    res_n = db(i, 1);
    scaled_input = (db(i, 2:785) / 255. * 0.99) + 0.01;
    targets = 0.01 * ones(1, output_nodes);
    targets(res_n + 1) = 0.99;
    n = n.train(scaled_input, targets);
end
test_db = readmatrix("mnist_test.csv");
true_res = [];
for i=1:length(test_db)
    res_n = test_db(i, 1);
    scaled_input = (test_db(i, 2:785) / 255. * 0.99) + 0.01;
    result = n.query(scaled_input);
    [v, id] = max(result);
    if id - 1 == res_n
        true_res(i) = 1;
    else
        true_res(i) = 0;
    end
    x = sprintf('Полученное %d - реальное %d', id - 1, res_n);
    disp(x);
end
s = sum(true_res, 'all');
l = length(true_res);
disp((s / l) * 100);

