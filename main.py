import random
import minerl
from minerl.viewer import get_parser

if __name__ == '__main__':
    # args = get_parser().parse_args()
    main(env_name="", stream_name="")
    parser = argparse.ArgumentParser(description='Visualize MineRL trajectories')
    parser.add_argument('--env-name', required=True,
                        help="MineRL environment, e.g. MineRLBasaltMakeWaterfall-v0")
    parser.add_argument('--sort', choices=sort_order_choices, default='up',
                        help='Animal sort order (default: %(default)s)')
    parser.add_argument('--uppercase', action='store_true',
                        help='Trajectory name, e.g. v3_accomplished_pattypan_squash_ghost-4_244-3238')


    print("Welcome to the MineRL Stream viewer! \n")

    print(f"Building data pipeline for {env_name}")
    data = minerl.data.make(env_name)

    print(f"Loading data for {stream_name}...")
    data_frames = list(data.load_data(stream_name, include_metadata=True))
    meta = data_frames[0][-1]
    # print("Data loading complete!")
    # print(f"META DATA: {meta}")
    # for frame in data_frames[:1]:
    #     print(frame)
    print(data_frames[0])