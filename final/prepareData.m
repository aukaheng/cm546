CSV = csvread('mnist/chinese_mnist_0-10.csv');
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

   % https://medium.com/analytics-vidhya/a-tip-a-day-python-tip-8-why-should-we-normalize-image-pixel-values-or-divide-by-255-4608ac5cd26a
  % Pixel values range from 0 to 256, excluding 0
  % imread result is an integer matrix
  M = double(M);
  M = M / 255;
  v = M(:);
  
  % Turn it into a row vector
  % Each row in X represents one image
  X(i, :) = v';

  y(i) = row(4);
endfor;

%fprintf('\nAll images is unrolled and loaded into a matrix with the size of %s.', num2str(size(images)));

save('-mat', 'final.mat', 'X', 'y');

%imshow(reshape(X(1,:), 64, 64));