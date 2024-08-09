#include "stratum.h"

namespace {
    inline static ethash::hash256 to_hash256(const std::string& hex)
    {
        std::string hex_string = hex;
        if(hex_string.substr(0, 2) == "0x")
            hex_string = hex_string.substr(2); //remove 0x
        auto parse_digit = [](char d) -> int { return d <= '9' ? (d - '0') : (d - 'a' + 10); };
        ethash::hash256 hash = {};
        for (size_t i = 1; i < hex_string.size(); i += 2)
        {
            int h = parse_digit(hex_string[i - 1]);
            int l = parse_digit(hex_string[i]);
            hash.bytes[i / 2] = static_cast<uint8_t>((h << 4) | l);
        }
        return hash;
    }

    inline static std::string to_string(const ethash::hash256& hash256)
    {
        constexpr auto hex_chars = "0123456789abcdef";
        std::string str;
        str.reserve(sizeof(hash256) * 2);
        for (const auto& b : hash256.bytes)
        {
            str.push_back(hex_chars[static_cast<uint8_t>(b) >> 4]);
            str.push_back(hex_chars[static_cast<uint8_t>(b) & 0xf]);
        }
        return "0x" + str;
    }
}

void Stratum::connect()
{
	char buf[max_length];
	asio::error_code error;
	//connection
	spdlog::info(prefix + "connecting to pool {} : {} ...", server, port);
	tcp::resolver resolver(io_context);
	asio::connect(socket, resolver.resolve(server, port));
	//pool authentication
	socket.write_some(asio::buffer(STRATUM_LOGIN));
	const size_t len = socket.read_some(asio::buffer(buf), error);
	if (error)
		throw asio::system_error(error);
	assert(len > 0 && len < max_length);
	std::string buffer(buf, len - 1);
	spdlog::debug("server reply {}", buffer);
	//login failed
	if (buffer.find("true") == std::string::npos)
		throw std::runtime_error("login failed, server reply: " + buffer);
}

void Stratum::listen()
{
	char buf[max_length];
	asio::error_code error;
    rapidjson::Document document;
    //pool get work
	socket.write_some(asio::buffer(STRATUM_ETH_GETWORK));
	while(true)
	{
		const size_t len = socket.read_some(asio::buffer(buf), error);
		if (error)
			throw asio::system_error(error);
		assert(len > 0);
        std::string buffer(buf, len - 1);
		spdlog::debug(prefix + "server reply: {}", buffer);
        const auto pos = buffer.find_last_of("\n");
        if(pos!= std::string::npos)
            buffer = buffer.substr(pos + 1);
		//parse server json return
        {
            document.Parse(buffer);
            if(document["id"].GetInt() != 0) {//other type of response
                spdlog::info(prefix + "server reply: {}", buffer);
                continue;
            }
            const rapidjson::Value& result = document["result"];
            std::string header     = result[0].GetString();
            std::string seed       = result[1].GetString();
            std::string difficulty = result[2].GetString();
            spdlog::debug(prefix + "header: {} seed: {} difficulty: {}", header, seed, difficulty);
            //update header
            if (stratumMiningConfig.header != header)
            {
                stratumMiningConfig.header = header;
                auto header_hash256 = to_hash256(header);
                for (auto &miner: miners)
                    miner->updateHeader(header_hash256);
            }
            //update seed & new epoch
            if (stratumMiningConfig.difficulty != difficulty)
            {
                stratumMiningConfig.difficulty = difficulty;
                auto difficulty_hash256 = to_hash256(difficulty);
                for (auto &miner: miners)
                    miner->updateDifficulty(difficulty_hash256);
            }
            //update seed & new epoch
            if (stratumMiningConfig.seed != seed)
            {
                stratumMiningConfig.seed = seed;
                auto seed_hash256 = to_hash256(seed);
                for (auto &miner: miners)
                {
                    miner->updateSeed(seed_hash256);
                    //first time start
                    if (miner->getMinerStatus() == VulkanEthminer::status::STOP)
                        std::thread{&VulkanEthminer::execute, miner}.detach();
                        //stop the current run, re-init epoch
                    else
                    {
                        spdlog::critical(prefix + "epoch changing...");
                        miner->setMinerStatus(VulkanEthminer::status::STOP);
                        std::this_thread::sleep_for(std::chrono::milliseconds(10000));
                        std::thread{&VulkanEthminer::execute, miner}.detach();
                    }
                }
            }
        }
	}
}

void Stratum::submit(solution solution, int deviceIndex){
    //nonce to string
    std::stringstream stream;
    stream << std::setfill('0') << std::setw(sizeof(uint64_t) * 2) << std::hex << solution.nonce;
    std::string nonce = stream.str();
    //mix to string
    const auto& ethash_result = ethash::hash(*solution.context, solution.header, solution.nonce);
    std::string finalHash = to_string(ethash_result.final_hash);
    //not meet the difficulty
    std::string difficulty = to_string(solution.difficulty);
    if (finalHash > difficulty)
        return;
    const std::string STRATUM_SUBMIT_WORK = "{ \"id\":1,\"method\" : \"eth_submitWork\",\"params\" : [\"0x" + nonce + "\",\"" + stratumMiningConfig.header + "\",\"" + to_string(ethash_result.mix_hash) + "\"] ,\"worker\" : \"" + rig +"\" }\n";
	socket.write_some(asio::buffer(STRATUM_SUBMIT_WORK));
    spdlog::critical("solution = (nonce: 0x{}, final_hash: {})", nonce, finalHash);
}
