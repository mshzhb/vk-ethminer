#pragma once
class Stratum;

#include <cassert>
#include <asio.hpp>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <spdlog/spdlog.h>
#include <rapidjson/document.h>
#include "../miner/vulkanminer.h"

using asio::ip::tcp;

class Stratum
{
public:
	//buffer length
	enum { max_length = 1024 };
    //log prefix
    const std::string prefix = "[" + cyan + "stratum" + reset + "] ";
	//pool cfg
	std::string server;
	std::string port;
	std::string wallet;
    std::string rig;
	//boost socket
	asio::io_context io_context;
	tcp::socket socket { io_context };
	//stratum message
	std::string STRATUM_LOGIN;
	std::string STRATUM_ETH_GETWORK;
    //miner list
    std::vector<VulkanEthminer*> miners;
    //mining info
    struct stratumMiningConfig{
        std::string header;
        std::string seed;
        std::string difficulty;
    };
    struct stratumMiningConfig stratumMiningConfig;

    struct solution {
        const ethash::epoch_context* context;
        uint64_t nonce;
        ethash::hash256 header;
        ethash::hash256 difficulty;
    };

	Stratum(std::string server, std::string port, std::string wallet, std::string rig)
	{
        std::transform(wallet.begin(), wallet.end(), wallet.begin(),[](unsigned char c) { return std::tolower(c); });
		this->server = server;
		this->port = port;
		this->wallet = wallet;
        this->rig = rig;
		//build stratum message
		STRATUM_LOGIN = "{ \"id\": 1, \"jsonrpc\" : \"2.0\", \"method\" : \"eth_submitLogin\", \"params\" : [\"" + wallet + "\"] }\n";
		STRATUM_ETH_GETWORK = "{ \"id\": 1, \"jsonrpc\": \"2.0\", \"method\": \"eth_getWork\" }\n";
	}

	void connect();
    void listen();
    void attachMiners(std::vector<VulkanEthminer*> miners){
        this->miners = miners;
    }


    void submit(solution solution, int gpuIndex);
};
