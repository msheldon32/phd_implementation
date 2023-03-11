addpath(genpath("/home/matts/imperial-qore-line-solver-3d80cf0/"));

rng(100);

n_classes = 3;
n_queues = 6;
max_class_size = 10;
n_repetitions = 16;
class_jump = 2;

for rep=1:n_repetitions
    get_table(rep,n_classes,n_queues,max_class_size, class_jump)
end

function get_table(table_id, n_classes, n_queues, max_class_size, class_jump)
    service_rates = get_service_rates(n_classes, n_queues);
    P = get_transition_matrix(n_classes, n_queues);

    populations = get_all_populations(n_classes, max_class_size, class_jump);

    output_arr = zeros([size(populations), 2]);
    for k = 1:size(populations,1)
        processed_network = process_network(n_classes, n_queues, service_rates, P, populations(k,:));
        for class_n = 1:n_classes
            output_arr(k,class_n) = populations(k, class_n);
            output_arr(k,n_classes+class_n) = processed_network{class_n};
        end
    end
    csvwrite(sprintf('output_%d.csv', table_id), output_arr);
end

function populations = get_all_populations(n_classes, max_class_size, class_jump)
    if n_classes == 1
        populations = reshape(1:class_jump:max_class_size,[length(1:class_jump:max_class_size),1]);
    else
        prev = get_all_populations(n_classes-1, max_class_size, class_jump);
        populations = [];
        for new_ct = 1:class_jump:max_class_size
            populations = cat(1, populations, cat(2,prev,ones(size(prev,1),1)*new_ct));
        end
    end
end

function service_rates = get_service_rates(n_classes, n_queues)
    min_service = 1;
    max_service = 20;

    service_rates = {};

    for queue_n = 1:n_queues
        service_rates{queue_n} = {};
        for class_n = 1:n_classes
            rate = (rand()* (max_service-min_service))+min_service;
            service_rates{queue_n}{class_n} = rate;
        end
    end
end

function P = get_transition_matrix(n_classes, n_queues)
    P = {};
    for class_n = 1:n_classes
        P{class_n} = zeros(n_queues, n_queues)
        for start_q = 1:n_queues
            out_nums = {};
            denom = 0;
            for end_q = 1:n_queues
                out_nums{end_q} = rand();
                denom = denom + out_nums{end_q};
            end

            for end_q = 1:n_queues
                P{class_n}(start_q, end_q) = out_nums{end_q}/denom;
            end
        end
    end
end

function [out_net, classes] = generate_network(n_classes, n_queues, service_rates, P, class_counts)
    out_net = Network ('qn');

    queues = {};
    classes = {};

    for queue_n = 1:n_queues
        queues {queue_n} = Queue(out_net, sprintf('Queue %d', queue_n), SchedStrategy.PS);
    end

    for class_n = 1:n_classes
        classes {class_n} = ClosedClass(out_net, sprintf('Class %d', class_n),class_counts(class_n),queues{1});
    end

    for queue_n = 1:n_queues
        for class_n = 1:n_classes
            queues{queue_n}.setService(classes{class_n}, Exp(service_rates{queue_n}{class_n}));
        end
    end

    out_net.link(P);
end

function tputs = process_network(n_classes, n_queues, service_rates, P, class_counts)
    [network, classes] = generate_network(n_classes, n_queues, service_rates, P, class_counts);
    for class_n = 1:length(class_counts)
        network.classes{class_n}.population = class_counts(class_n);
    end
    soln_table = SolverMVA(network, 'method', 'exact').getAvgTable;
    tputs = {};
    for class_n = 1:length(class_counts)
        tputs{class_n} = sum(soln_table(soln_table.JobClass == classes{class_n}.name,:).Tput);
    end
end