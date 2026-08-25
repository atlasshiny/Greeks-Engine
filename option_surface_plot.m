% Select the binary file dynamically
[filename, pathname] = uigetfile('*.bin', 'Select dataset');
if isequal(filename, 0), return; end
full_path = fullfile(pathname, filename);

% Open file and inspect raw bytes
fileID = fopen(full_path, 'r');
fileInfo = dir(full_path);
totalBytes = fileInfo.bytes;

% Standard C++ float size is 4 bytes
bytesPerElement = 4;
totalFloats = totalBytes / bytesPerElement;

% Check for standard 100x100 grid or dynamically derive grid side length
% If header bytes exist, strip them dynamically to find the largest square payload
possibleSide = floor(sqrt(totalFloats));
gridSize = possibleSide; 
headerElements = mod(totalFloats, gridSize * gridSize);

% If header accounts for non-square remainder, skip header elements
if headerElements > 0
    fseek(fileID, headerElements * bytesPerElement, 'bof');
end

% Read matrix dynamically as 32-bit floating point ('single')
Z = fread(fileID, [gridSize, gridSize], 'single')';
fclose(fileID);

% Render Surface Plot
figure;
surf(Z);
colorbar;
colormap('jet');
shading interp;

xlabel('Volatility Index');
ylabel('Spot Price Index');
zlabel('Option Price ($)');
title(sprintf('Dynamic Option Surface (%dx%d)', gridSize, gridSize), 'Interpreter', 'none');
grid on;