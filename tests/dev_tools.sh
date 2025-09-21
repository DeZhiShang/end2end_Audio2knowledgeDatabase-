#!/bin/bash
"""
开发工具脚本
提供开发过程中常用的快捷命令
"""

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
    echo -e "  ${GREEN}help${NC}      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 clean     # 清理所有中间文件和结果文件"
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

# 主函数
main() {
    case "${1:-help}" in
        "clean")
            clean_files
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