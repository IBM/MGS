function [Voltage Calcium]=plotter(path,V,Ca,f,hld,clr)
colors=['kgmbrcy'];
Voltage=cell(size(V,1),1);
Calcium=cell(size(Ca,1),1);
figure(f);
for i=1:size(V,1)
    %% Initialize variables.

    filename = [path '/rec' num2str(V(i,1)) '.dat' num2str(V(i,2))]
    delimiter = '\t';
    startRow = 2;

    %% Read columns of data as strings:
    % For more information, see the TEXTSCAN documentation.
    formatSpec = ['%s'];
    for j=1:V(i,3)-1
        formatSpec=[formatSpec '%*s'];
    end
    formatSpec=[formatSpec '%s'];
    formatSpec=[formatSpec '%[^\n\r]'];

    %% Open the text file.
    fileID = fopen(filename,'r');

    %% Read columns of data according to format string.
    % This call is based on the structure of the file used to generate this
    % code. If an error occurs for a different file, try regenerating the code
    % from the Import Tool.
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);

    %% Close the text file.
    fclose(fileID);

    %% Convert the contents of columns containing numeric strings to numbers.
    % Replace non-numeric strings with NaN.
    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = dataArray{col};
    end
    numericData = NaN(size(dataArray{1},1),size(dataArray,2));

    for col=[1, 2]
        % Converts strings in the input cell array to numbers. Replaced non-numeric
        % strings with NaN.
        rawData = dataArray{col};
        for row=1:size(rawData, 1);
            % Create a regular expression to detect and remove non-numeric prefixes and
            % suffixes.
            regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
            try
                result = regexp(rawData{row}, regexstr, 'names');
                numbers = result.numbers;

                % Detected commas in non-thousand locations.
                invalidThousandsSeparator = false;
                if any(numbers==',');
                    thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                    if isempty(regexp(thousandsRegExp, ',', 'once'));
                        numbers = NaN;
                        invalidThousandsSeparator = true;
                    end
                end
                % Convert numeric strings to numbers.
                if ~invalidThousandsSeparator;
                    numbers = textscan(strrep(numbers, ',', ''), '%f');
                    numericData(row, col) = numbers{1};
                    raw{row, col} = numbers{1};
                end
            catch me
            end
        end
    end


    %% Replace non-numeric cells with NaN
    R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
    raw(R) = {NaN}; % Replace non-numeric cells

    %% Allocate imported array to column variable names
    Time = cell2mat(raw(:, 1));    
    Voltage{i,1} = cell2mat(raw(:, 2));


    %% Clear temporary variables
    clearvars filename delimiter startRow formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me R;

    subplot(2,1,1);
    if (i==1 && hld==0) hold off; 
    else hold on;
    end;
    plot(Time,Voltage{i},colors(mod(i+clr,size(colors,2))));
    ylabel('V');
end;



for i=1:size(Ca,1)
    %% Initialize variables.

    filename = [path '/rec' num2str(Ca(i,1)) '.dat' num2str(Ca(i,2))]
    delimiter = '\t';
    startRow = 2;

    %% Read columns of data as strings:
    % For more information, see the TEXTSCAN documentation.
    formatSpec = ['%s'];
    for j=1:Ca( i,3)-1
        formatSpec=[formatSpec '%*s'];
    end
    formatSpec=[formatSpec '%s'];
    formatSpec=[formatSpec '%[^\n\r]'];

    %% Open the text file.
    fileID = fopen(filename,'r');

    %% Read columns of data according to format string.
    % This call is based on the structure of the file used to generate this
    % code. If an error occurs for a different file, try regenerating the code
    % from the Import Tool.
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);

    %% Close the text file.
    fclose(fileID);

    %% Convert the contents of columns containing numeric strings to numbers.
    % Replace non-numeric strings with NaN.
    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = dataArray{col};
    end
    numericData = NaN(size(dataArray{1},1),size(dataArray,2));

    for col=[1, 2]
        % Converts strings in the input cell array to numbers. Replaced non-numeric
        % strings with NaN.
        rawData = dataArray{col};
        for row=1:size(rawData, 1);
            % Create a regular expression to detect and remove non-numeric prefixes and
            % suffixes.
            regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
            try
                result = regexp(rawData{row}, regexstr, 'names');
                numbers = result.numbers;

                % Detected commas in non-thousand locations.
                invalidThousandsSeparator = false;
                if any(numbers==',');
                    thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                    if isempty(regexp(thousandsRegExp, ',', 'once'));
                        numbers = NaN;
                        invalidThousandsSeparator = true;
                    end
                end
                % Convert numeric strings to numbers.
                if ~invalidThousandsSeparator;
                    numbers = textscan(strrep(numbers, ',', ''), '%f');
                    numericData(row, col) = numbers{1};
                    raw{row, col} = numbers{1};
                end
            catch me
            end
        end
    end


    %% Replace non-numeric cells with NaN
    R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
    raw(R) = {NaN}; % Replace non-numeric cells

    %% Allocate imported array to column variable names
    Time = cell2mat(raw(:, 1));
    Calcium{i,1} = cell2mat(raw(:, 2));


    %% Clear temporary variables
    clearvars filename delimiter startRow formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me R;

    subplot(2,1,2);
    if (i==1 && hld==0) hold off; 
    else hold on;
    end;
    plot(Time,Calcium{i},colors(mod(i+clr,size(colors,2))));
    ylabel('[Ca2+]');
    xlabel('msec');
end;
set(findall(gcf,'type','axes'),'fontSize',20);
set(findall(gcf,'type','text'),'fontSize',25);
