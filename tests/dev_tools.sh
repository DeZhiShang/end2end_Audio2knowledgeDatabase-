#!/bin/bash
#
# 开发工具脚本
# 提供开发过程中常用的快捷命令
#

# 设置脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}端到端音频处理系统 - 开发工具${NC}"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "可用命令:"
    echo -e "  ${GREEN}clean${NC}     删除所有处理过程中产生的中间文件和结果文件"
    echo -e "  ${GREEN}rename${NC}    将MP3文件按自增序重命名为规范格式"
    echo -e "  ${GREEN}help${NC}      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 clean     # 清理所有中间文件和结果文件"
    echo "  $0 rename    # 重命名MP3文件为001.mp3, 002.mp3, ..."
    echo ""
}

# 清理处理过程中产生的文件
clean_files() {
    echo -e "${YELLOW}🧹 开始清理处理过程中产生的文件...${NC}"

    # 切换到项目根目录
    cd "$PROJECT_ROOT"

    # 定义要清理的目录
    DIRS_TO_CLEAN=(
        "data/output/docs"
        "data/processed/rttms"
        "data/processed/wavs"
    )

    # 统计删除的文件数量
    total_files=0

    for dir in "${DIRS_TO_CLEAN[@]}"; do
        if [ -d "$dir" ]; then
            # 计算目录中的文件数量（包括子目录中的文件）
            file_count=$(find "$dir" -type f | wc -l)

            if [ "$file_count" -gt 0 ]; then
                echo -e "${BLUE}📁 清理目录: $dir${NC}"
                echo -e "   发现 $file_count 个文件"

                # 删除目录中的所有文件，保留目录结构
                find "$dir" -type f -delete

                # 删除空的子目录，但保留主目录
                find "$dir" -mindepth 1 -type d -empty -delete

                echo -e "   ${GREEN}✅ 已删除 $file_count 个文件${NC}"
                total_files=$((total_files + file_count))
            else
                echo -e "${BLUE}📁 目录已清空: $dir${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  目录不存在: $dir${NC}"
        fi
    done

    # 清理日志文件（可选）
    if [ -d "logs" ]; then
        log_count=$(find logs -name "*.log" -o -name "*.json" | wc -l)
        if [ "$log_count" -gt 0 ]; then
            echo -e "${BLUE}📁 清理日志目录: logs${NC}"
            echo -e "   发现 $log_count 个日志文件"
            find logs -name "*.log" -delete
            find logs -name "*.json" -delete
            echo -e "   ${GREEN}✅ 已删除 $log_count 个日志文件${NC}"
            total_files=$((total_files + log_count))
        fi
    fi

    echo ""
    echo -e "${GREEN}🎉 清理完成！总共删除了 $total_files 个文件${NC}"
    echo -e "${BLUE}💡 提示: 目录结构已保留，可以直接运行处理流程${NC}"
}

# 重命名MP3文件为自增序格式
rename_mp3_files() {
    echo -e "${YELLOW}🔄 开始重命名MP3文件...${NC}"

    # 切换到项目根目录
    cd "$PROJECT_ROOT"

    # 定义MP3文件目录
    MP3_DIR="data/input/batch_2"

    # 检查目录是否存在
    if [ ! -d "$MP3_DIR" ]; then
        echo -e "${RED}❌ 错误: 目录不存在: $MP3_DIR${NC}"
        return 1
    fi

    # 检查是否有MP3文件
    mp3_count=$(find "$MP3_DIR" -name "*.mp3" -type f | wc -l)
    if [ "$mp3_count" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  目录中没有找到MP3文件${NC}"
        return 0
    fi

    echo -e "${BLUE}📁 在目录 $MP3_DIR 中找到 $mp3_count 个MP3文件${NC}"

    # 询问用户确认
    echo -e "${YELLOW}⚠️  此操作将把所有MP3文件重命名为 001.mp3, 002.mp3, ... 格式${NC}"
    echo -e "${YELLOW}⚠️  原始文件名将被永久改变${NC}"
    echo -n "是否继续? (y/N): "
    read -r confirm

    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}💡 操作已取消${NC}"
        return 0
    fi

    # 创建临时目录存储重命名映射
    temp_dir=$(mktemp -d)
    counter=1

    echo -e "${BLUE}🔄 开始重命名处理...${NC}"

    # 获取所有MP3文件并排序
    readarray -t mp3_files < <(find "$MP3_DIR" -name "*.mp3" -type f | sort)

    # 遍历文件进行重命名
    for file in "${mp3_files[@]}"; do
        if [ -f "$file" ]; then
            # 生成新的文件名（3位数字格式）
            new_name=$(printf "%03d.mp3" $counter)
            temp_file="$temp_dir/$new_name"

            # 移动文件到临时目录
            mv "$file" "$temp_file"
            echo "   $(basename "$file") -> $new_name"

            ((counter++))
        fi
    done

    # 将文件从临时目录移回原目录
    echo -e "${BLUE}📦 完成重命名，移回原目录...${NC}"
    if [ -n "$(ls -A "$temp_dir" 2>/dev/null)" ]; then
        mv "$temp_dir"/*.mp3 "$MP3_DIR/" 2>/dev/null
    fi

    # 清理临时目录
    rmdir "$temp_dir" 2>/dev/null

    # 统计最终结果
    final_count=$(find "$MP3_DIR" -name "*.mp3" -type f | wc -l)

    echo ""
    echo -e "${GREEN}🎉 重命名完成！${NC}"
    echo -e "${GREEN}✅ 成功重命名了 $final_count 个MP3文件${NC}"
    echo -e "${BLUE}💡 文件现在按 001.mp3, 002.mp3, ... 格式命名${NC}"
}

# 主函数
main() {
    case "${1:-help}" in
        "clean")
            clean_files
            ;;
        "rename")
            rename_mp3_files
            ;;
        "help" | "-h" | "--help")
            show_help
            ;;
        *)
            echo -e "${RED}❌ 未知命令: $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"