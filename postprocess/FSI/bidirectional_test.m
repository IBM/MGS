numSynapses = size(rows,2);
%% Cherck if rows and columns are the right size
for i=1:numSynapses
    if (size(rows{i}) ~= size(cols{i}))
        disp('BAD');
    end
end

%% Check if number of GJs is around 9
num = zeros(1,numSynapses);
for i=1:numSynapses
    num(i) = numel(rows{i});
end
figure(1); clf;
histogram(num);
mean(num)

%% Check for only one connection to any neuron and not itself
for i=1:numSynapses
    if (numel(unique(cols{i})) ~= numel(cols{i}))
        disp('More than one connection to a neuron');
        return
    end
    if (sum(ismember(cols{i},i)) > 0)
        disp('Connecting to itself.');
    end
end

%% Check for only bidirectional
for i=1:numSynapses
    for j=1:numel(cols{i})
        if (sum(ismember(cols{i},cols{i}(j))) ~= 1)
            disp('BAD 2');
        end
    end
end

%% Put in to a matrix
con = false(27*27*27, 27*27*27);
for i=1:numSynapses
    for j=1:numel(cols{i})
        con(rows{i}(j), cols{i}(j)) = true;
    end
end
issymmetric(con)

figure(2); clf;
histogram(sum(con));