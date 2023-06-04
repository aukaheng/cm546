CSV = csvread('mnist/chinese_mnist.csv');
fprintf('\nLoaded the index mapping file.\n');

% Remove the header 1:1
% Remove the Chinese character E:E
CSV = CSV(2:end, 1:4);

m = size(CSV, 1);

% Each image size is 64x64 pixels
inputLayerSize = 64 * 64;

X = zeros(m, inputLayerSize);
y = zeros(m, 1);

for i = 1:m
  row = CSV(i, :);

  fileName = [num2str(row(:, 1)), '_', num2str(row(:, 2)), '_', num2str(row(:, 3))];

  fprintf('Reading file %s.jpg\n', fileName);
  fprintf('Its value is %d\n\n', row(4));

  % A matrix representation of the image
  % 64x64
  M = imread(['mnist/data/data/input_', fileName , '.jpg']);
  %fprintf('\nSize of matrix is %s.\n', num2str(size(matrix)));

  % Turn the matrix into a row vector
  % "reshape" performs a column wise scan, so make sure to first transpose the matrix
  v = reshape(M', [1, inputLayerSize]);
  %fprintf('\nSize of vector is %s.\n', num2str(size(vector)));

  X(i, :) = v;
  y(i) = row(4);
endfor;

%fprintf('\nAll images is unrolled and loaded into a matrix with the size of %s.', num2str(size(images)));

save('-mat', 'final.mat', 'X', 'y');