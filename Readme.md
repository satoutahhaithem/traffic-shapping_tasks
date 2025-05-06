## netstat -i | grep $INTERFACE Command Definition
This command displays network interface statistics for a specific network interface:

## netstat -i: Shows statistics for all network interfaces (packets received, transmitted, errors, etc.)
|: Pipes the output to the next command
grep $INTERFACE: Filters the output to show only information about the specific interface stored in the $INTERFACE variable
This is commonly used in network scripts to monitor or troubleshoot a particular network interface's performance metrics.

## tc -s qdisc show dev $INTERFACE
This command displays traffic control queuing discipline configurations and statistics:

## tc: Linux Traffic Control command for managing network traffic
-s: Shows detailed statistics
qdisc show: Displays the configured queuing disciplines (scheduling algorithms)
dev $INTERFACE: Specifies the network interface stored in the $INTERFACE variable
This command is used for monitoring and troubleshooting Quality of Service (QoS) settings and network traffic management configurations on a specific interface.