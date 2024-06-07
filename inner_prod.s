	.def	_ZN131_$LT$he_ring..rnsconv..approx_lift..AlmostExactBaseConversion$LT$R$C$M_Int$C$M_Zn$GT$$u20$as$u20$he_ring..rnsconv..RNSOperation$GT$5apply28_$u7b$$u7b$closure$u7d$$u7d$23actuall_compute_product17hb478fb896be209b8E;
	.scl	3;
	.type	32;
	.endef
	.section	.text,"xr",one_only,_ZN131_$LT$he_ring..rnsconv..approx_lift..AlmostExactBaseConversion$LT$R$C$M_Int$C$M_Zn$GT$$u20$as$u20$he_ring..rnsconv..RNSOperation$GT$5apply28_$u7b$$u7b$closure$u7d$$u7d$23actuall_compute_product17hb478fb896be209b8E
	.p2align	4, 0x90
_ZN131_$LT$he_ring..rnsconv..approx_lift..AlmostExactBaseConversion$LT$R$C$M_Int$C$M_Zn$GT$$u20$as$u20$he_ring..rnsconv..RNSOperation$GT$5apply28_$u7b$$u7b$closure$u7d$$u7d$23actuall_compute_product17hb478fb896be209b8E:
.Lfunc_begin404:
	.cv_func_id 86530
	.cv_loc	86530 16 196 0
.seh_proc _ZN131_$LT$he_ring..rnsconv..approx_lift..AlmostExactBaseConversion$LT$R$C$M_Int$C$M_Zn$GT$$u20$as$u20$he_ring..rnsconv..RNSOperation$GT$5apply28_$u7b$$u7b$closure$u7d$$u7d$23actuall_compute_product17hb478fb896be209b8E
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$152, %rsp
	.seh_stackalloc 152
	.seh_endprologue
.Ltmp99771:
	movq	%r9, %rsi
	movq	%r8, 104(%rsp)
.Ltmp99772:
	movq	%rdx, %rdi
	movq	%rcx, 96(%rsp)
.Ltmp99773:
	movq	256(%rsp), %rcx
.Ltmp99774:
	.cv_inline_site_id 86531 within 86530 inlined_at 16 199 0
	.cv_inline_site_id 86532 within 86531 inlined_at 127 32 0
	.cv_inline_site_id 86533 within 86532 inlined_at 7 318 0
	.cv_inline_site_id 86534 within 86533 inlined_at 7 279 0
	.cv_inline_site_id 86535 within 86534 inlined_at 19 154 0
	.cv_inline_site_id 86536 within 86535 inlined_at 7 245 0
	.cv_loc	86536 19 527 0
	movq	32(%rcx), %rax
.Ltmp99775:
	leaq	(%rax,%rax), %rdx
.Ltmp99776:
	movq	%rdx, 80(%rsp)
	movq	%rax, 88(%rsp)
.Ltmp99777:
	.cv_inline_site_id 86537 within 86536 inlined_at 19 527 0
	.cv_loc	86537 7 500 0
	testl	$1073741824, %eax
	je	.LBB404_3
.Ltmp99778:
	.cv_inline_site_id 86538 within 86536 inlined_at 19 527 0
	.cv_loc	86538 7 495 0
	movq	80(%rsp), %rax
.Ltmp99779:
	mulq	%rax
.Ltmp99780:
	.cv_loc	86538 7 496 0
	cmpq	$2147483647, %rax
	sbbq	$0, %rdx
.Ltmp99781:
	movl	$2147483647, %edx
.Ltmp99782:
	.cv_loc	86537 7 506 0
	cmovll	%eax, %edx
	testl	%edx, %edx
	jns	.LBB404_3
.Ltmp99783:
	.cv_loc	86537 7 516 0
	cmpq	$0, 16(%rcx)
	je	.LBB404_31
.Ltmp99784:
.LBB404_3:
	.cv_inline_site_id 86539 within 86533 inlined_at 7 280 0
	.cv_inline_site_id 86540 within 86539 inlined_at 1 108 0
	.cv_inline_site_id 86541 within 86540 inlined_at 2 47 0
	.cv_inline_site_id 86542 within 86541 inlined_at 1 108 0
	.cv_inline_site_id 86543 within 86542 inlined_at 6 1075 0
	.cv_inline_site_id 86544 within 86543 inlined_at 8 305 0
	.cv_inline_site_id 86545 within 86544 inlined_at 7 318 0
	.cv_inline_site_id 86546 within 86545 inlined_at 7 303 0
	.cv_inline_site_id 86547 within 86546 inlined_at 1 108 0
	.cv_inline_site_id 86548 within 86547 inlined_at 1 108 0
	.cv_inline_site_id 86549 within 86548 inlined_at 6 1075 0
	.cv_inline_site_id 86550 within 86549 inlined_at 8 305 0
	.cv_loc	86550 16 200 0
	movabsq	$9223372036854775807, %r12
	andq	88(%rsp), %r12
	decq	%r12
	movq	(%rcx), %rax
	movq	%rax, 64(%rsp)
	movq	8(%rcx), %rax
	movq	%rax, 56(%rsp)
	movq	16(%rcx), %rax
	movq	%rax, 48(%rsp)
.Ltmp99785:
	leaq	-1(%rdi), %rax
	movq	%rax, 112(%rsp)
	movq	104(%rsp), %rax
.Ltmp99786:
	.cv_loc	86540 2 44 0
	addq	$8, %rax
	movq	%rax, 136(%rsp)
	movq	96(%rsp), %rax
	addq	$8, %rax
	movq	%rax, 128(%rsp)
	xorl	%ebx, %ebx
	xorl	%eax, %eax
	movq	%r12, 72(%rsp)
	jmp	.LBB404_4
.Ltmp99787:
	.p2align	4, 0x90
.LBB404_24:
	movq	72(%rsp), %r12
	movb	$1, %r9b
.Ltmp99788:
.LBB404_29:
	.cv_inline_site_id 86551 within 86533 inlined_at 7 285 0
	.cv_inline_site_id 86552 within 86551 inlined_at 7 137 0
	.cv_loc	86552 7 28 0
	movq	%r14, %rax
	movq	56(%rsp), %r8
	mulq	%r8
	movq	%rdx, %rcx
.Ltmp99789:
	.cv_loc	86552 7 28 0
	movq	%r15, %rax
	mulq	64(%rsp)
.Ltmp99790:
	.cv_inline_site_id 86553 within 86552 inlined_at 7 32 0
	.cv_loc	86553 15 1817 0
	imulq	%r8, %r15
.Ltmp99791:
	.cv_loc	86551 7 137 0
	addq	%rcx, %rdx
	addq	%r15, %rdx
.Ltmp99792:
	.cv_inline_site_id 86554 within 86551 inlined_at 7 138 0
	.cv_inline_site_id 86555 within 86554 inlined_at 7 32 0
	.cv_loc	86555 15 1817 0
	imulq	48(%rsp), %rdx
.Ltmp99793:
	.cv_loc	86554 15 1794 0
	subq	%rdx, %r14
.Ltmp99794:
	.cv_inline_site_id 86556 within 86533 inlined_at 7 285 0
	.cv_loc	86556 7 230 0
	addq	120(%rsp), %r14
.Ltmp99795:
	movq	80(%rsp), %rax
.Ltmp99796:
	.cv_inline_site_id 86557 within 86556 inlined_at 7 230 0
	.cv_loc	86557 7 167 0
	cmpq	%rax, %r14
	movl	$0, %ecx
	cmovbq	%rcx, %rax
	subq	%rax, %r14
.Ltmp99797:
	movq	88(%rsp), %rax
	.cv_loc	86557 7 170 0
	cmpq	%rax, %r14
	cmovbq	%rcx, %rax
	subq	%rax, %r14
.Ltmp99798:
	movq	%r14, %rax
.Ltmp99799:
	.cv_loc	86540 2 44 0
	testb	%r9b, %r9b
	je	.LBB404_30
.Ltmp99800:
.LBB404_4:
	.cv_inline_site_id 86558 within 86547 inlined_at 1 108 0
	.cv_inline_site_id 86559 within 86558 inlined_at 4 844 0
	.cv_inline_site_id 86560 within 86559 inlined_at 4 753 0
	.cv_loc	86560 9 1565 0
	cmpq	%rdi, %rbx
.Ltmp99801:
	.cv_loc	86559 4 753 0
	jae	.LBB404_5
.Ltmp99802:
	.cv_loc	86550 16 200 0
	movq	%rax, 120(%rsp)
.Ltmp99803:
	movq	112(%rsp), %r8
.Ltmp99804:
	subq	%rbx, %r8
	movq	104(%rsp), %rax
	leaq	(%rax,%rbx,8), %r10
	movq	96(%rsp), %rax
	leaq	(%rax,%rbx,8), %r11
	xorl	%eax, %eax
	xorl	%r9d, %r9d
	xorl	%r14d, %r14d
.Ltmp99805:
	.p2align	4, 0x90
.LBB404_7:
	.cv_inline_site_id 86561 within 86559 inlined_at 4 756 0
	.cv_inline_site_id 86562 within 86561 inlined_at 4 208 0
	.cv_loc	86562 15 533 0
	leaq	(%rbx,%rax), %rcx
.Ltmp99806:
	.cv_loc	86550 16 200 0
	cmpq	%rdi, %rcx
	jae	.LBB404_32
.Ltmp99807:
	cmpq	%rsi, %rcx
	jae	.LBB404_33
.Ltmp99808:
	.cv_inline_site_id 86563 within 86546 inlined_at 1 108 0
	.cv_loc	86563 6 1075 0
	movq	%rax, %r15
.Ltmp99809:
	movq	(%r11,%rax,8), %rax
.Ltmp99810:
	.cv_loc	86545 7 307 0
	mulq	(%r10,%r15,8)
	addq	%rax, %r9
.Ltmp99811:
	adcq	%rdx, %r14
.Ltmp99812:
	.cv_inline_site_id 86564 within 86545 inlined_at 7 302 0
	.cv_inline_site_id 86565 within 86564 inlined_at 4 844 0
	.cv_inline_site_id 86566 within 86565 inlined_at 4 753 0
	.cv_loc	86566 9 1565 0
	cmpq	$31, %r15
.Ltmp99813:
	.cv_loc	86565 4 753 0
	je	.LBB404_11
.Ltmp99814:
	leaq	1(%r15), %rax
	cmpq	%r15, %r8
	jne	.LBB404_7
.Ltmp99815:
.LBB404_11:
	.cv_loc	86545 7 315 0
	addq	%r15, %rbx
	incq	%rbx
	movq	%r14, 32(%rsp)
	movq	64(%rsp), %rcx
	movq	56(%rsp), %rdx
	movq	48(%rsp), %r8
	callq	_ZN11feanor_math5rings2zn5zn_646ZnBase21bounded_reduce_larger17hd4a6098608cf2ac7E
	movq	%rax, %r14
.Ltmp99816:
	movb	$1, %r9b
.Ltmp99817:
	xorl	%r15d, %r15d
	testq	%r12, %r12
.Ltmp99818:
	.cv_inline_site_id 86567 within 86533 inlined_at 7 282 0
	.cv_loc	86567 153 39 0
	jne	.LBB404_12
	jmp	.LBB404_29
.Ltmp99819:
	.p2align	4, 0x90
.LBB404_22:
	movq	%rax, %rbx
.Ltmp99820:
.LBB404_23:
	.cv_inline_site_id 86568 within 86567 inlined_at 153 41 0
	.cv_inline_site_id 86569 within 86568 inlined_at 34 4109 0
	.cv_inline_site_id 86570 within 86569 inlined_at 1 108 0
	.cv_inline_site_id 86571 within 86570 inlined_at 2 47 0
	.cv_inline_site_id 86572 within 86571 inlined_at 1 108 0
	.cv_inline_site_id 86573 within 86572 inlined_at 6 1075 0
	.cv_inline_site_id 86574 within 86573 inlined_at 8 305 0
	.cv_inline_site_id 86575 within 86574 inlined_at 7 318 0
	.cv_loc	86575 7 315 0
	movq	%r10, 32(%rsp)
	movq	64(%rsp), %rcx
	movq	56(%rsp), %rdx
	movq	48(%rsp), %r8
	callq	_ZN11feanor_math5rings2zn5zn_646ZnBase21bounded_reduce_larger17hd4a6098608cf2ac7E
.Ltmp99821:
	.cv_loc	86533 7 283 0
	addq	%rax, %r14
.Ltmp99822:
	adcq	$0, %r15
.Ltmp99823:
	.cv_loc	86567 153 39 0
	decq	%r12
.Ltmp99824:
	je	.LBB404_24
.Ltmp99825:
.LBB404_12:
	.cv_inline_site_id 86576 within 86575 inlined_at 7 302 0
	.cv_inline_site_id 86577 within 86576 inlined_at 4 844 0
	.cv_loc	86577 4 753 0
	leaq	32(%rbx), %rax
	movq	%rax, 144(%rsp)
	movq	136(%rsp), %rax
	leaq	(%rax,%rbx,8), %rbp
	movq	128(%rsp), %rax
	leaq	(%rax,%rbx,8), %r13
	xorl	%r8d, %r8d
	xorl	%ecx, %ecx
	xorl	%r11d, %r11d
.Ltmp99826:
	.p2align	4, 0x90
.LBB404_13:
	.cv_inline_site_id 86578 within 86575 inlined_at 7 303 0
	.cv_inline_site_id 86579 within 86578 inlined_at 1 108 0
	.cv_inline_site_id 86580 within 86579 inlined_at 1 108 0
	.cv_inline_site_id 86581 within 86580 inlined_at 4 844 0
	.cv_inline_site_id 86582 within 86581 inlined_at 4 753 0
	.cv_loc	86582 9 1565 0
	leaq	(%rbx,%r11), %rax
	cmpq	%rdi, %rax
.Ltmp99827:
	.cv_loc	86581 4 753 0
	jae	.LBB404_14
.Ltmp99828:
	.cv_inline_site_id 86583 within 86579 inlined_at 1 108 0
	.cv_inline_site_id 86584 within 86583 inlined_at 6 1075 0
	.cv_inline_site_id 86585 within 86584 inlined_at 8 305 0
	.cv_loc	86585 16 200 0
	cmpq	%rsi, %rax
	jae	.LBB404_26
.Ltmp99829:
	.cv_inline_site_id 86586 within 86578 inlined_at 1 108 0
	.cv_loc	86586 6 1075 0
	movq	-8(%r13,%r11,8), %rax
.Ltmp99830:
	.cv_loc	86575 7 307 0
	mulq	-8(%rbp,%r11,8)
	movq	%rax, %r9
	movq	%rdx, %r10
.Ltmp99831:
	.cv_loc	86585 16 200 0
	leaq	(%rbx,%r11), %rax
	incq	%rax
.Ltmp99832:
	.cv_loc	86575 7 307 0
	addq	%r8, %r9
.Ltmp99833:
	adcq	%rcx, %r10
.Ltmp99834:
	.cv_loc	86582 9 1565 0
	cmpq	%rdi, %rax
.Ltmp99835:
	.cv_loc	86581 4 753 0
	jae	.LBB404_22
.Ltmp99836:
	.cv_loc	86585 16 200 0
	cmpq	%rsi, %rax
	jae	.LBB404_25
.Ltmp99837:
	.cv_loc	86586 6 1075 0
	movq	(%r13,%r11,8), %rax
.Ltmp99838:
	.cv_loc	86575 7 307 0
	mulq	(%rbp,%r11,8)
	movq	%rax, %r8
	movq	%rdx, %rcx
.Ltmp99839:
	leaq	2(%r11), %rax
.Ltmp99840:
	addq	%r9, %r8
.Ltmp99841:
	adcq	%r10, %rcx
.Ltmp99842:
	movq	%rax, %r11
.Ltmp99843:
	.cv_inline_site_id 86587 within 86577 inlined_at 4 753 0
	.cv_loc	86587 9 1565 0
	cmpq	$32, %rax
.Ltmp99844:
	.cv_loc	86577 4 753 0
	jne	.LBB404_13
.Ltmp99845:
	movq	144(%rsp), %rbx
	jmp	.LBB404_16
.Ltmp99846:
	.p2align	4, 0x90
.LBB404_14:
	.cv_loc	86575 7 309 0
	testq	%r11, %r11
	je	.LBB404_28
.Ltmp99847:
	movq	%rax, %rbx
.Ltmp99848:
.LBB404_16:
	movq	%r8, %r9
	movq	%rcx, %r10
	jmp	.LBB404_23
.Ltmp99849:
.LBB404_28:
	xorl	%r9d, %r9d
	movq	%rax, %rbx
	movq	72(%rsp), %r12
	jmp	.LBB404_29
.Ltmp99850:
.LBB404_5:
	movq	%rax, %r14
.Ltmp99851:
.LBB404_30:
	.cv_loc	86530 16 202 0
	movq	%r14, %rax
	addq	$152, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
.Ltmp99852:
	popq	%rsi
.Ltmp99853:
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq
.Ltmp99854:
.LBB404_26:
	.cv_loc	86585 16 200 0
	addq	%r11, %rbx
.Ltmp99855:
	leaq	alloc_58fa99bead828d0a394cfb3ec457b765(%rip), %r8
.Ltmp99856:
	movq	%rbx, %rcx
.Ltmp99857:
	movq	%rsi, %rdx
	callq	_ZN4core9panicking18panic_bounds_check17h030234dcd63f312fE
.Ltmp99858:
.LBB404_25:
	addq	%r11, %rbx
	incq	%rbx
.Ltmp99859:
	leaq	alloc_58fa99bead828d0a394cfb3ec457b765(%rip), %r8
	movq	%rbx, %rcx
	movq	%rsi, %rdx
	callq	_ZN4core9panicking18panic_bounds_check17h030234dcd63f312fE
.Ltmp99860:
.LBB404_33:
	.cv_loc	86550 16 200 0
	leaq	alloc_58fa99bead828d0a394cfb3ec457b765(%rip), %r8
	movq	%rsi, %rdx
	callq	_ZN4core9panicking18panic_bounds_check17h030234dcd63f312fE
.Ltmp99861:
.LBB404_32:
	leaq	alloc_1b4af8cd7935e97d0b05a15a13975456(%rip), %r8
	movq	%rdi, %rdx
	callq	_ZN4core9panicking18panic_bounds_check17h030234dcd63f312fE
.Ltmp99862:
.LBB404_31:
	.cv_loc	86537 7 516 0
	leaq	alloc_baa6253b6718be5b1b06d82c092be485(%rip), %rcx
.Ltmp99863:
	callq	_ZN4core9panicking11panic_const23panic_const_rem_by_zero17h2da8143d8048d37aE
	int3
.Ltmp99864:
.Lfunc_end404:
	.seh_endproc