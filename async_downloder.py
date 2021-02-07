import sys
import asyncio
import aiohttp
import aiofiles

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def fetch(session, filename, url, proxy=None):
    async with session.get(url, proxy=proxy) as resp:
        # Only proceed further if the HTTP response is 200 (Ok)
        if resp.status == 200:
            async with aiofiles.open(filename, mode='wb') as f:
                await f.write(await resp.read())
                await f.close()

async def main(image_urls:dict, timeout=20, proxy_type="http", proxy=None):
    tasks = []
    if proxy is not None:
        proxy = "{}//{}".format(proxy_type, proxy)

    timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for filename, image in image_urls.items():
            tasks.append(fetch(session, filename, image, proxy))
        data = await asyncio.gather(*tasks)

if __name__ == '__main__':

    #filename: image_url
    single_image = 'https://understandingdata.com/wp-content/uploads/2019/09/james-anthony-phoenix.jpg'
    images = {'base1.jpg':single_image}
    asyncio.run(main(images))