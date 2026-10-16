
mkdir ./pdfs

python3 gen_dot_sub.py


function change_format()
{
    raw_name=$1
    out_name=$2
    index=$3
    final_out_name=${index}.${out_name}
    cat ./posts/$raw_name.md.raw | python3 ./posts/change_format_pdf.py > ./pdfs/${final_out_name}-pdf.md
    cat ./posts/$raw_name.md.raw | python3 ./posts/change_format_md.py > ./posts/${final_out_name}.md
    cd pdfs
    pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./${final_out_name}-pdf.md -o ./${out_name}.pdf
    cd -
}

change_format pre_llm pre_llm 1.1
change_format llm_intro llm_intro 1.2
change_format llm_archs llm_archs 1.3
change_format llm_sft_and_usages llm_sft_and_usages 1.4
change_format llm_alignment llm_alignment 1.5
change_format llm_multimodal llm_multimodal 1.6
change_format llm_recommend llm_recommend 1.7
change_format llm_o1 llm_o1 1.8
change_format llm_others llm_others 1.9

change_format recommend recommend 2
change_format llm_rec_summary llm_rec_summary 3
change_format llm_note llm_note 4
change_format llm_agentic llm_agentic 5
change_format llm_math llm_math 6
change_format int-ml int-ml 8
change_format full collections 9
change_format bitter_lesson bitter_lesson 00
change_format ai_videos ai_videos 01


rm -rf pdfs/*.md

rm -rf ../daiwk_llm_pdfs/*
mv ./pdfs/* ~/Downloads/同步空间/daiwk_llm_pdfs/
rm -rf ./pdfs

git add ./assets
git add posts
git add gen_pdfs.sh
git add gen_dot_sub.py
git add publish
git add deep-research-assets
git add notes 
git commit -m 'x'
git push

