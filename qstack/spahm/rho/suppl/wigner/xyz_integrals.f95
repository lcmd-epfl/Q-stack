program xyz_integrals
  implicit none
  integer :: argc, n,m,k
  character(len=32) :: argv(1:3)
  argc = command_argument_count()
  if (argc<3) stop
  do n = 1, 3
    call get_command_argument(n, argv(n))
  enddo
  read (argv(1),'(i4)') n
  read (argv(2),'(i4)') m
  read (argv(3),'(i4)') k
  write(*,'(f17.15,A)') xyz(n,m,k), " π"
contains

real*8 function xyz(n,m,k)
  integer, value :: n,m,k
  call sort3 (k,n,m)
  ! k>=n>=m
  if (n==0) then
    xyz = 2.0 * (1.0 - (2*k-1)/dble(2*k+1))
  else
    !xyz = (2*k-1) * I2(n,m,k) * I3(n,m)
    xyz = (2*k-1) * I23(n,m,k)
  endif
  return
end function xyz

real*8 function I2(n,m,k)
  integer, intent(in) :: n,m,k
  integer :: l
  I2 = 0.0
  do l = 0, n+m+1
    I2 = I2 + (-1)**l * binomial(n+m+1, l) / dble(2*l+2*k-1)
  enddo
  return
end function I2

real*8 function I3(n,m)
  integer, intent(in) :: n,m
  integer :: l
  I3 = binomial(2*(n+m+1), n+m+1) / dble( (2*n+1) * 2**(2*n+2*m) )
  do l = 1, n+1
    I3 = I3 * (2*n+3-2*l) / dble(2*m-1+2*l)
  enddo
  return
end function I3

real*8 function I23(n,m,k)
  integer, intent(in) :: n,m,k
  integer :: l
  I23 = 0.0
  do l = 0, n+m+1
    I23 = I23 + (-1)**l * trinomial( n+m+1, n+m+1-l, l) / dble(2*l+2*k-1);
  enddo
  I23 = I23 / dble( (2*n+1) * 2**(2*n+2*m) )
  do l = 1, n+1
    I23 = I23 * (2*n+3-2*l) / dble(2*m-1+2*l)
  enddo
  return
end function I23

!-------------------------------------------------------------------------------

real*8 function binomial (n, k)
  integer, value :: n,k
  integer :: i

  if (k<0 .or. k>n) then
    binomial = 0.0
    return
  endif

  binomial = 1.0
  if (k==0 .or. k==n) return

  if (k>n-k) k=n-k
  do i=1, k
    binomial = binomial * (n-i+1)/dble(i)
  enddo

  return
end function binomial

real*8 function trinomial (k1, k2, k3)
  !  (k1+k2+k3)! / (k1! * k2! * k3!)
  integer, value :: k1,k2,k3
  integer :: k
  call sort3(k1,k2,k3)
  trinomial = 1.0
  do k = 1, k2
    trinomial = trinomial * (k+k1) / dble(k)
  enddo
  do k = 1, k3
    trinomial = trinomial * (k+k1+k2) / dble(k)
  enddo
  return
end function trinomial

subroutine swap (i, j)
  integer, intent(inout) :: i,j
  integer :: t
  t = i; i = j; j = t
end subroutine swap

subroutine sort3 (k1, k2, k3)
  integer, intent(inout) :: k1,k2,k3
  if (k1<k2) call swap(k1,k2)
  if (k1<k3) call swap(k1,k3)
  if (k2<k3) call swap(k2,k3)
end subroutine sort3

end program xyz_integrals

