#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
词汇表查询工具 (Vocabulary Lookup Tool)

功能:
1. 输入一个词语，查询它在词汇表文件中的索引 (ID)。
2. 输入一个索引 (ID)，查询它对应的词语。

使用方法:
- 查询词语: python vocab_lookup.py --file [词汇表文件路径] --word [要查询的词语]
- 查询索引: python vocab_lookup.py --file [词汇表文件路径] --index [要查询的索引]

例如:
python vocab_lookup.py --file clip_disect_20k.txt --word cat
python vocab_lookup.py --file clip_disect_20k.txt --index 1760
"""

import argparse
import sys

def main():
    """主函数，负责解析参数和执行查询。"""
    
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="一个用来查询词汇表文件索引和词语的便捷工具。",
        epilog="示例: python vocab_lookup.py --file clip_disect_20k.txt --word cat"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="必须提供！词汇表文件的路径 (例如: ../models/cache/.../clip_disect_20k.txt)"
    )
    
    # 创建一个互斥参数组，确保用户要么查词，要么查索引，不能同时进行
    lookup_group = parser.add_mutually_exclusive_group(required=True)
    
    lookup_group.add_argument(
        "--word",
        "-w",
        type=str,
        help="要查询的词语 (例如: cat, car, dog)。"
    )
    
    lookup_group.add_argument(
        "--index",
        "-i",
        type=int,
        help="要查询的索引/ID (例如: 1760)。"
    )
    
    args = parser.parse_args()

    # --- 2. 读取和处理词汇表文件 ---
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            # 读取所有行，并去除每行末尾的换行符，存入列表
            # 列表的索引天然就是我们需要的 ID
            vocab_list = [line.strip() for line in f]
        
        print(f"✅ 成功加载词汇表: {args.file} (共 {len(vocab_list)} 个词语)")
        print("-" * 30)

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{args.file}'。", file=sys.stderr)
        print("请检查 --file 参数提供的路径是否正确。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取文件时发生未知错误: {e}", file=sys.stderr)
        sys.exit(1)


    # --- 3. 执行查询逻辑 ---
    if args.word:
        # 用户提供了 --word 参数，执行“词语 -> 索引”查询
        try:
            target_word = args.word.lower() # 统一转为小写以增加匹配成功率
            index = vocab_list.index(target_word)
            print(f"🔍 查找【词语】: '{args.word}'")
            print(f"🎉 结果: 索引 (ID) 是 👉 {index}")
            
        except ValueError:
            print(f"🔍 查找【词语】: '{args.word}'")
            print(f"🤷‍♂️ 结果: 在词汇表中未找到该词语。")
            
    elif args.index is not None:
        # 用户提供了 --index 参数，执行“索引 -> 词语”查询
        try:
            word = vocab_list[args.index]
            print(f"🔍 查找【索引】: {args.index}")
            print(f"🎉 结果: 对应的词语是 👉 '{word}'")
            
        except IndexError:
            print(f"🔍 查找【索引】: {args.index}")
            print(f"🤷‍♂️ 结果: 索引越界。有效索引范围是 0 到 {len(vocab_list) - 1}。")

if __name__ == "__main__":
    main()